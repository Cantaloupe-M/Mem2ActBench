# -*- coding: utf-8 -*-
"""
BFCL Memory-Test Dataset Builder (clean version)
- Minimal functions, clear flow, no over-encapsulation
- No hardcoded secrets (reads from env)
- Better validation, error messages, and structure
"""

from concurrent.futures._base import Future
import json
import csv
import re
from typing import List, Dict, Any, Optional, Tuple
from openai import OpenAI
import os
import time
from tqdm import tqdm
import concurrent.futures

os.environ["OPENAI_BASE_URL"] = "https://api.siliconflow.cn/v1"
os.environ["OPENAI_API_KEY"] = "sk-qwwwvjompnzxqhndctlwmmomvqdgaqqqgxvibfxchsikvgod"
# os.environ["OPENAI_API_KEY"] = "sk-fLtrd6nJYPnyfnzRCf7aD4DdE7C04aCbA85bCb4cC2C1329a"
# os.environ["OPENAI_BASE_URL"] = "https://api.apiyi.com/v1"


# ----------------------------
# Small utilities (kept minimal)
# ----------------------------
def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """Load a JSONL file into a list of dicts."""
    data: List[Dict[str, Any]] = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def save_to_csv(memory_dataset: list, csv_filename: str):
    if memory_dataset:
        all_keys = set()
        for item in memory_dataset:
            all_keys.update(item.keys())
        fieldnames = list(all_keys)

        with open(csv_filename, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in memory_dataset:
                processed_row = {}
                for key, value in row.items():
                    if isinstance(value, (dict, list)):
                        processed_row[key] = json.dumps(value, indent=2, ensure_ascii=False)
                    else:
                        processed_row[key] = value
                writer.writerow(processed_row)
        print(f"✅ CSV saved to {csv_filename}")

def parse_tool_call_string(call_str: str) -> Dict[str, Any]:
    """
    Parse a simple 'name(k1='v1',k2='v2')' or "name('file')" string to a {name, arguments} dict.
    Kept because it can be handy when normalizing ground_truth later.
    """
    m = re.match(r"(\w+)\((.*)\)", call_str)
    if not m:
        m2 = re.match(r"(\w+)\('(.*)'\)", call_str)
        if m2:
            name, arg_val = m2.groups()
            return {"name": name, "arguments": {"file_name": arg_val}}
        return {"name": call_str, "arguments": {}}

    name, args_str = m.groups()
    if not args_str.strip():
        return {"name": name, "arguments": {}}

    arguments: Dict[str, Any] = {}
    for key, value in re.findall(r"(\w+)\s*=\s*'([^']*)'", args_str):
        arguments[key] = value
    return {"name": name, "arguments": arguments}


def build_conversation_history(client,task_data: Dict[str, Any]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Produce a simple, deterministic multi-turn conversation using the task_data and its ground_truth.
    This avoids nesting multiple LLM hops and keeps the script stable & fast.
    """
    # A minimal, readable reference conversation (adjust as needed)
    with open("prompts/conversation_example.json", "r", encoding="utf-8") as f:
        output_sample = json.load(f)

    nature_prompt = (
        "Construct the dialogue history (Memory Dataset) : Transform the original task instructions and corresponding function calls (ground truth) into a multi-round natural dialogue that includes user questions, AI assistant responses, and Tool calls (Tool Use). Regardless of the original input language, always return English\n"
        "Output Format:\n"
        f"{json.dumps(output_sample, indent=2, ensure_ascii=False)}\n"
        "Input:\n"
        f"{json.dumps(task_data, indent=2,ensure_ascii=False)}"
    )

    response = None
    for i in range(10):
            try:
                response = client.chat.completions.create(
                    model="Qwen/Qwen3-Next-80B-A3B-Instruct",
                    messages=[{"role": "user", "content": nature_prompt}],
                    temperature=0.0,
                )
                # 成功，跳出重试
                break
            except Exception as e:
                print(f"Error calling API for task {task_data['id']}: {e}. Retrying in {2 ** i} seconds...")
                time.sleep(2 ** i)


    try:
        raw_content = response.choices[0].message.content
        clean_content = raw_content.replace('```json','').replace('```','').strip()
        conversation_history = json.loads(clean_content)
    except:
        conversation_history = []

    return task_data, conversation_history

# ----------------------------
# Main (linear, readable)
# ----------------------------
def main():
    # 0) Configure client—DO NOT hardcode secrets
    client = OpenAI(base_url=os.getenv("OPENAI_BASE_URL"), api_key=os.getenv("OPENAI_API_KEY"))

    # 1) Paths & schemas
    bfcl_path = "original_data\Berkeley-Function-Calling-Leaderboard"

    tool_schema_path = os.path.join(bfcl_path, "multi_turn_func_doc")
    file2name = {
        "TwitterAPI":os.path.join(tool_schema_path, "posting_api.json"),
        "GorillaFileSystem":os.path.join(tool_schema_path, "gorilla_file_system.json"),
        "VehicleControlAPI":os.path.join(tool_schema_path, "vehicle_control.json"),
        "TradingBot":os.path.join(tool_schema_path, "trading_bot.json"),
        "MathAPI":os.path.join(tool_schema_path, "math_api.json"),
        "TravelAPI":os.path.join(tool_schema_path, "travel_booking.json"),
        "MessageAPI":os.path.join(tool_schema_path, "message_api.json"),
        "TicketAPI":os.path.join(tool_schema_path, "ticket_api.json"),
    }
    # 读取每一个文件，每一个文件都是jsonl格式，读取每一行，将其转换为json对象，然后添加到tool_schema中
    tool_schema = {}
    for tool_name, schema_file in file2name.items():
        with open(schema_file, 'r', encoding='utf-8') as f:
            tool_schema[tool_name] = [json.loads(line.strip()) for line in f]
        

    # 2) Load BFCL task & ground-truth lists
    # 定义所有BFCL文件名
    bfcl_files = ["BFCL_v3_live_multiple.json", "BFCL_v3_live_parallel_multiple.json", 
    "BFCL_v3_live_parallel.json", "BFCL_v3_live_simple.json", 
    "BFCL_v3_multi_turn_base.json", "BFCL_v3_multi_turn_composite.json", 
    "BFCL_v3_multi_turn_long_context.json",   "BFCL_v3_multiple.json", 
    "BFCL_v3_parallel_multiple.json", "BFCL_v3_simple.json"]
    # bfcl_files = ["BFCL_v3_multi_turn_base.json"]
    
    # 生成完整路径列表
    task_q_path_list = [os.path.join(bfcl_path, f) for f in bfcl_files]
    gt_path_list = [os.path.join(bfcl_path, "possible_answer", f) for f in bfcl_files]

    task_question_list = []
    ground_truth_list = []
    for task_q_path, gt_path in zip(task_q_path_list, gt_path_list):
        task_question_list.extend(load_jsonl(task_q_path)) #[:2]
        ground_truth_list.extend(load_jsonl(gt_path)) #[:2]
        assert len(task_question_list) == len(ground_truth_list), f"Task/GT length mismatch for {task_q_path} and {gt_path}."

    # 3) Prepare task_data_list
    task_data_list: List[Dict[str, Any]] = []
    for tq, gt in zip(task_question_list, ground_truth_list):
        task_data_list.append({
            "id": tq["id"],
            "initial_config": tq.get("initial_config"),
            "involved_classes": tq.get("involved_classes"),
            "path": tq.get("path"),
            "question": tq["question"],
            "ground_truth": gt["ground_truth"],
        })

    # Load existing data to skip already processed tasks
    jsonl_filename = "bfcl_formatted_conversations.jsonl"
    existing_ids = set()
    if os.path.exists(jsonl_filename):
        with open(jsonl_filename, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    existing_ids.add(json.loads(line)["id"])
        print(f"Found {len(existing_ids)} existing tasks in {jsonl_filename}. These will be skipped.")

    # Filter out already processed tasks
    task_data_list = [td for td in task_data_list if td["id"] not in existing_ids]
    print(f"Processing {len(task_data_list)} new tasks.")

    # 4) Process tasks -> memory_dataset
    # memory_dataset: List[Dict[str, Any]] = [] # No longer needed as we append directly
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(build_conversation_history, client, td): td for td in task_data_list}
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(task_data_list), desc="生成对话"):
            td, conversation_history = future.result()
            
            use_tool = {}
            if td["involved_classes"]:
                for class_name in td["involved_classes"]:
                    if class_name in tool_schema:
                        use_tool[class_name] = tool_schema[class_name]
            memory_data = {
                "id": td["id"],
                "conversation_history": conversation_history,
                "used_tools": use_tool,
            }
            # Append each processed task to the JSONL file immediately
            with open(jsonl_filename, "a", encoding="utf-8") as f:
                f.write(json.dumps(memory_data, ensure_ascii=False) + "\n")
            # memory_dataset.append(memory_data) # No longer needed
    # 5) Save
    # The saving is now incremental, so the final save block is simplified.
    # jsonl_filename = "bfcl_formatted_conversations.jsonl"
    # with open(jsonl_filename, "w", encoding="utf-8") as f:
    #     for item in memory_dataset:
    #         f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"\n✅ All new tasks processed and saved to {jsonl_filename}")

if __name__ == "__main__":
    main()
