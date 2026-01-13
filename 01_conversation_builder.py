# -*- coding: utf-8 -*-
"""
Unified Conversation Dataset Builder
- Supports BFCL, OASST1, and ToolACE datasets
- CLI-driven with configurable parameters
- No hardcoded secrets (reads from env or CLI)
"""

import argparse
import json
import os
import re
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI
from tqdm import tqdm


# ----------------------------
# Common Utilities
# ----------------------------
def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """Load a JSONL file into a list of dicts."""
    data: List[Dict[str, Any]] = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def load_existing_ids(jsonl_path: str) -> set:
    """Load existing IDs from output file to enable resume."""
    if not os.path.exists(jsonl_path):
        return set()
    ids = set()
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                ids.add(json.loads(line)["id"])
    return ids


def append_jsonl(file_path: str, record: Dict[str, Any]):
    """Append a single record to JSONL file."""
    with open(file_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def call_llm_with_retry(
    client: OpenAI,
    prompt: str,
    model: str,
    temperature: float = 0.0,
    max_retries: int = 10,
) -> Optional[str]:
    """Call LLM with exponential backoff retry."""
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
            )
            return response.choices[0].message.content
        except Exception as e:
            if attempt < max_retries - 1:
                delay = 2 ** attempt
                print(f"API error (attempt {attempt + 1}/{max_retries}): {e}. Retrying in {delay}s...")
                time.sleep(delay)
            else:
                print(f"API failed after {max_retries} attempts: {e}")
                return None
    return None


def parse_llm_json_response(raw: str) -> Any:
    """Parse JSON from LLM response, stripping markdown fences."""
    cleaned = raw.replace("```json", "").replace("```", "").strip()
    return json.loads(cleaned)


# ----------------------------
# BFCL Processing
# ----------------------------
class BFCLProcessor:
    """Process Berkeley Function Calling Leaderboard dataset."""

    TOOL_SCHEMA_FILES = {
        "TwitterAPI": "posting_api.json",
        "GorillaFileSystem": "gorilla_file_system.json",
        "VehicleControlAPI": "vehicle_control.json",
        "TradingBot": "trading_bot.json",
        "MathAPI": "math_api.json",
        "TravelAPI": "travel_booking.json",
        "MessageAPI": "message_api.json",
        "TicketAPI": "ticket_api.json",
    }

    BFCL_FILES = [
        "BFCL_v3_live_multiple.json",
        "BFCL_v3_live_parallel_multiple.json",
        "BFCL_v3_live_parallel.json",
        "BFCL_v3_live_simple.json",
        "BFCL_v3_multi_turn_base.json",
        "BFCL_v3_multi_turn_composite.json",
        "BFCL_v3_multi_turn_long_context.json",
        "BFCL_v3_multiple.json",
        "BFCL_v3_parallel_multiple.json",
        "BFCL_v3_simple.json",
    ]

    def __init__(self, data_path: str, example_path: str):
        self.data_path = data_path
        self.example_path = example_path
        self.tool_schema = self._load_tool_schemas()

    def _load_tool_schemas(self) -> Dict[str, List[Dict]]:
        """Load tool schemas from files."""
        schema_dir = os.path.join(self.data_path, "multi_turn_func_doc")
        tool_schema = {}
        for tool_name, filename in self.TOOL_SCHEMA_FILES.items():
            schema_file = os.path.join(schema_dir, filename)
            if os.path.exists(schema_file):
                tool_schema[tool_name] = load_jsonl(schema_file)
        return tool_schema

    def load_tasks(self) -> List[Dict[str, Any]]:
        """Load and merge task questions with ground truth."""
        task_list, gt_list = [], []
        for filename in self.BFCL_FILES:
            task_path = os.path.join(self.data_path, filename)
            gt_path = os.path.join(self.data_path, "possible_answer", filename)
            if os.path.exists(task_path) and os.path.exists(gt_path):
                task_list.extend(load_jsonl(task_path))
                gt_list.extend(load_jsonl(gt_path))

        tasks = []
        for tq, gt in zip(task_list, gt_list):
            tasks.append({
                "id": tq["id"],
                "initial_config": tq.get("initial_config"),
                "involved_classes": tq.get("involved_classes"),
                "path": tq.get("path"),
                "question": tq["question"],
                "ground_truth": gt["ground_truth"],
            })
        return tasks

    def build_prompt(self, task_data: Dict[str, Any]) -> str:
        """Build LLM prompt for conversation generation."""
        with open(self.example_path, "r", encoding="utf-8") as f:
            example = json.load(f)
        return (
            "Construct the dialogue history (Memory Dataset): Transform the original task "
            "instructions and corresponding function calls (ground truth) into a multi-round "
            "natural dialogue that includes user questions, AI assistant responses, and Tool "
            "calls (Tool Use). Regardless of the original input language, always return English\n"
            f"Output Format:\n{json.dumps(example, indent=2, ensure_ascii=False)}\n"
            f"Input:\n{json.dumps(task_data, indent=2, ensure_ascii=False)}"
        )

    def build_record(self, task_data: Dict, conversation: List) -> Dict[str, Any]:
        """Build final record with tools."""
        use_tools = {}
        if task_data.get("involved_classes"):
            for cls in task_data["involved_classes"]:
                if cls in self.tool_schema:
                    use_tools[cls] = self.tool_schema[cls]
        return {
            "id": task_data["id"],
            "conversation_history": conversation,
            "used_tools": use_tools,
        }


# ----------------------------
# OASST1 Processing
# ----------------------------
class OASST1Processor:
    """Process Open Assistant dataset."""

    def __init__(self, data_path: str):
        self.data_path = data_path

    def load_tasks(self) -> List[Dict[str, Any]]:
        """Load and deduplicate conversations."""
        import pandas as pd

        records = load_jsonl(self.data_path)
        df = pd.DataFrame(records)
        msg_by_id = df.set_index("message_id").to_dict("index")
        best_assist = df.query("role == 'assistant' and rank == 0").copy()

        conversations = []
        for _, row in best_assist.iterrows():
            path = []
            cur = row.to_dict()
            while cur is not None:
                turn = {
                    "role": "user" if cur.get("role") == "prompter" else "assistant",
                    "content": cur.get("text", ""),
                }
                path.append(turn)
                pid = cur.get("parent_id")
                cur = msg_by_id.get(pid)
            conversations.append(list(reversed(path)))

        # Deduplicate by first prompt, keep longest
        prompt2convs = defaultdict(list)
        for conv in conversations:
            prompt2convs[conv[0]["content"]].append(conv)
        deduped = [max(convs, key=len) for convs in prompt2convs.values()]

        return [{"id": f"oss_{i}", "conversation": conv} for i, conv in enumerate(deduped)]

    def translate_conversation(self, conv: List[Dict]) -> List[Dict]:
        """Translate conversation to English."""
        from deep_translator import GoogleTranslator

        for turn in conv:
            try:
                turn["content"] = GoogleTranslator(source="auto", target="en").translate(
                    turn["content"]
                )
            except Exception as e:
                print(f"Translation failed: {e}")
        return conv

    def build_record(self, task: Dict, conversation: List) -> Dict[str, Any]:
        """Build final record."""
        return {"id": task["id"], "conversation_history": conversation}


# ----------------------------
# ToolACE Processing
# ----------------------------
class ToolACEProcessor:
    """Process ToolACE dataset."""

    def __init__(self, data_path: str):
        self.data_path = data_path

    def load_tasks(self, start_idx: int = 0, end_idx: int = 8000) -> List[Dict[str, Any]]:
        """Load ToolACE data from local JSON."""
        with open(self.data_path, encoding="utf-8") as f:
            raw_dataset = json.load(f)

        tasks = []
        for idx, item in enumerate(raw_dataset):
            if not (start_idx <= idx < end_idx):
                continue
            rid = f"toolace_{idx}"
            use_tools = self._extract_tools(item.get("system", ""))
            conv = self._convert_conversations(item.get("conversations", []))
            tasks.append({"id": rid, "conversation": conv, "use_tools": use_tools})
        return tasks

    def _extract_tools(self, system_text: str) -> Optional[Dict]:
        """Extract tools from system prompt."""
        for line in system_text.splitlines():
            if '"name":' in line:
                for attempt in [line, line[:-1], line[:-2]]:
                    try:
                        return json.loads(attempt)
                    except json.JSONDecodeError:
                        continue
        return None

    def _convert_conversations(self, conversations: List[Dict]) -> List[Dict[str, Any]]:
        """Convert ToolACE format to standard format."""
        output = []
        call_counter = 0
        pending_calls = []

        for msg in conversations:
            role = msg.get("from")
            value = msg.get("value", "")

            if role == "user":
                output.append({"role": "user", "content": value})

            elif role == "assistant":
                content, blocks = self._strip_brackets(value)
                entry = {"role": "assistant", "content": content or ""}
                tool_calls = []

                for block in blocks:
                    for call_str in self._split_calls(block):
                        fname, fargs = self._parse_call(call_str)
                        call_counter += 1
                        call_id = f"call_{call_counter}"
                        tool_calls.append({
                            "id": call_id,
                            "type": "function",
                            "function": {"name": fname, "arguments": fargs},
                        })
                        pending_calls.append((call_id, fname))

                if tool_calls:
                    entry["tool_calls"] = tool_calls
                output.append(entry)

            elif role == "tool":
                results = self._parse_tool_results(value)
                for res in results:
                    if pending_calls:
                        call_id, fname = pending_calls.pop(0)
                    else:
                        call_id, fname = None, res.get("name", "tool")
                    payload = res.get("results")
                    output.append({
                        "role": "tool",
                        "tool_call_id": call_id,
                        "name": res.get("name", fname),
                        "content": payload if isinstance(payload, str) else json.dumps(payload),
                    })
        return output

    def _strip_brackets(self, text: str) -> Tuple[str, List[str]]:
        """Extract bracket blocks from text."""
        if not text:
            return "", []
        start = text.find("[")
        if start == -1:
            return text.strip(), []

        depth, end = 0, -1
        for j, ch in enumerate(text[start:], start=start):
            if ch == "[":
                depth += 1
            elif ch == "]":
                depth -= 1
                if depth == 0:
                    end = j
                    break

        if end == -1:
            return text.strip(), []
        block = text[start + 1 : end].strip()
        remaining = (text[:start] + text[end + 1 :]).strip()
        return remaining, [block]

    def _split_calls(self, block: str) -> List[str]:
        """Split function calls at top-level commas."""
        calls, buf, depth, in_str, str_q = [], [], 0, False, None

        for ch in block:
            buf.append(ch)
            if in_str:
                if ch == str_q:
                    in_str, str_q = False, None
            else:
                if ch in ('"', "'"):
                    in_str, str_q = True, ch
                elif ch == "(":
                    depth += 1
                elif ch == ")":
                    depth = max(0, depth - 1)
                elif ch == "," and depth == 0:
                    buf.pop()
                    s = "".join(buf).strip()
                    if s:
                        calls.append(s)
                    buf.clear()

        s = "".join(buf).strip()
        if s:
            calls.append(s)
        return calls

    def _parse_call(self, call_str: str) -> Tuple[str, str]:
        """Parse function call string."""
        m = re.match(r"\s*(.+?)\s*\(", call_str)
        if not m:
            return call_str.strip(), "{}"

        name = " ".join(m.group(1).split())
        start = call_str.find("(")
        depth, end = 0, -1
        for i, ch in enumerate(call_str[start:], start=start):
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0:
                    end = i
                    break

        args = call_str[start + 1 : end].strip() if end != -1 else ""
        return name, json.dumps(args) if args else "{}"

    def _parse_tool_results(self, value: str) -> List[Dict]:
        """Parse tool results from value."""
        try:
            data = json.loads(value)
            if isinstance(data, list):
                return data
            if isinstance(data, dict):
                return [data]
            return [{"name": "tool", "results": data}]
        except Exception:
            return [{"name": "tool", "results": value}]

    def build_prompt(self, conv: List[Dict]) -> str:
        """Build LLM prompt for conversation refinement."""
        example = self._get_example()
        return (
            "Construct the dialogue history (Memory Dataset): Transform the original task "
            "instructions and corresponding function calls (ground truth) into a multi-round "
            "natural dialogue that includes user questions, AI assistant responses, and Tool calls.\n"
            f"Output Format:\n{json.dumps(example, indent=2, ensure_ascii=False)}\n"
            f"Input:\n{json.dumps(conv, indent=2, ensure_ascii=False)}"
        )

    def _get_example(self) -> List[Dict]:
        """Return example conversation format."""
        return [
            {"role": "user", "content": "Hello! Please move `final_report.pdf` into a new `temp` folder."},
            {
                "role": "assistant",
                "content": "Sure — creating folder and moving the file now.",
                "tool_calls": [
                    {"id": "call_1", "type": "function", "function": {"name": "mkdir", "arguments": '{"dir_name": "temp"}'}},
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "name": "mkdir", "content": '{"status": "success"}'},
        ]

    def build_record(self, task: Dict, conversation: List) -> Dict[str, Any]:
        """Build final record."""
        return {
            "id": task["id"],
            "conversation_history": conversation,
            "use_tools": task.get("use_tools"),
        }


# ----------------------------
# Main Entry Point
# ----------------------------
def process_bfcl(args, client: OpenAI):
    """Process BFCL dataset."""
    processor = BFCLProcessor(args.data_path, args.example_path)
    tasks = processor.load_tasks()
    existing_ids = load_existing_ids(args.output)
    tasks = [t for t in tasks if t["id"] not in existing_ids]
    print(f"Processing {len(tasks)} BFCL tasks (skipped {len(existing_ids)} existing)")

    def process_one(task):
        prompt = processor.build_prompt(task)
        raw = call_llm_with_retry(client, prompt, args.model, args.temperature)
        if raw:
            try:
                conv = parse_llm_json_response(raw)
                return processor.build_record(task, conv)
            except json.JSONDecodeError:
                pass
        return processor.build_record(task, [])

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_one, t): t for t in tasks}
        for future in tqdm(as_completed(futures), total=len(tasks), desc="BFCL"):
            record = future.result()
            append_jsonl(args.output, record)


def process_oasst1(args, client: OpenAI):
    """Process OASST1 dataset."""
    processor = OASST1Processor(args.data_path)
    tasks = processor.load_tasks()
    existing_ids = load_existing_ids(args.output)
    tasks = [t for t in tasks if t["id"] not in existing_ids]
    print(f"Processing {len(tasks)} OASST1 conversations (skipped {len(existing_ids)} existing)")

    def process_one(task):
        conv = processor.translate_conversation(task["conversation"])
        return processor.build_record(task, conv)

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(process_one, t) for t in tasks]
        for future in tqdm(as_completed(futures), total=len(tasks), desc="OASST1"):
            record = future.result()
            append_jsonl(args.output, record)


def process_toolace(args, client: OpenAI):
    """Process ToolACE dataset."""
    processor = ToolACEProcessor(args.data_path)
    tasks = processor.load_tasks(args.start_idx, args.end_idx)
    existing_ids = load_existing_ids(args.output)
    tasks = [t for t in tasks if t["id"] not in existing_ids]
    print(f"Processing {len(tasks)} ToolACE tasks (skipped {len(existing_ids)} existing)")

    def process_one(task):
        prompt = processor.build_prompt(task["conversation"])
        raw = call_llm_with_retry(client, prompt, args.model, args.temperature)
        if raw:
            try:
                conv = parse_llm_json_response(raw)
                return processor.build_record(task, conv)
            except json.JSONDecodeError:
                pass
        return processor.build_record(task, [])

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_one, t): t for t in tasks}
        for future in tqdm(as_completed(futures), total=len(tasks), desc="ToolACE"):
            try:
                record = future.result()
                append_jsonl(args.output, record)
            except Exception as e:
                print(f"Error: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Unified Conversation Dataset Builder",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="dataset", required=True, help="Dataset to process")

    # Common arguments
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--api-base", default=os.getenv("OPENAI_BASE_URL"), help="OpenAI API base URL")
    common.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY"), help="OpenAI API key")
    common.add_argument("--model", default="Qwen/Qwen3-Next-80B-A3B-Instruct", help="Model name")
    common.add_argument("--temperature", type=float, default=0.0, help="Generation temperature")
    common.add_argument("--workers", type=int, default=4, help="Number of parallel workers")
    common.add_argument("-o", "--output", required=True, help="Output JSONL file path")

    # BFCL subcommand
    bfcl_parser = subparsers.add_parser("bfcl", parents=[common], help="Process BFCL dataset")
    bfcl_parser.add_argument(
        "--data-path",
        default="original_data/Berkeley-Function-Calling-Leaderboard",
        help="Path to BFCL data directory",
    )
    bfcl_parser.add_argument(
        "--example-path",
        default="prompts/conversation_example.json",
        help="Path to conversation example JSON",
    )

    # OASST1 subcommand
    oasst1_parser = subparsers.add_parser("oasst1", parents=[common], help="Process OASST1 dataset")
    oasst1_parser.add_argument(
        "--data-path",
        default="original_data/oasst1/2023-04-12_oasst_ready.messages.jsonl",
        help="Path to OASST1 JSONL file",
    )

    # ToolACE subcommand
    toolace_parser = subparsers.add_parser("toolace", parents=[common], help="Process ToolACE dataset")
    toolace_parser.add_argument(
        "--data-path",
        default="original_data/ToolACE/data.json",
        help="Path to ToolACE JSON file",
    )
    toolace_parser.add_argument("--start-idx", type=int, default=0, help="Start index for processing")
    toolace_parser.add_argument("--end-idx", type=int, default=8000, help="End index for processing")

    args = parser.parse_args()

    # Initialize client
    if not args.api_key:
        parser.error("API key required. Set OPENAI_API_KEY env var or use --api-key")

    client = OpenAI(base_url=args.api_base, api_key=args.api_key)

    # Dispatch to appropriate processor
    if args.dataset == "bfcl":
        process_bfcl(args, client)
    elif args.dataset == "oasst1":
        process_oasst1(args, client)
    elif args.dataset == "toolace":
        process_toolace(args, client)

    print(f"✅ Done! Output saved to {args.output}")


if __name__ == "__main__":
    main()
