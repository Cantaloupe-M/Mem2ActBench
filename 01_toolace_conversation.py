# -*- coding: utf-8 -*-
"""
ToolACE Memory-Test Dataset Builder
- Adapts the memory-test generation idea for the ToolACE dataset.
- Loads data directly from Hugging Face.
- Dynamically creates conversation history from the 'solution' trace.
"""
import os

os.environ["OPENAI_BASE_URL"] = "https://api.siliconflow.cn/v1"
os.environ["OPENAI_API_KEY"] = "sk-omutivisxktqoglinfimxbckcxacgaqayusvamjquihiujov"
# os.environ["OPENAI_API_KEY"] = "sk-fLtrd6nJYPnyfnzRCf7aD4DdE7C04aCbA85bCb4cC2C1329a"
# os.environ["OPENAI_BASE_URL"] = "https://api.apiyi.com/v1"

import json
import csv
import re
from typing import List, Dict, Any, Optional,Tuple
from tqdm import tqdm
from collections import defaultdict
from sentence_transformers import SentenceTransformer, util
import time
from openai import OpenAI
# Removed 'from datasets import load_dataset' as per editing plan

def save_to_csv(memory_dataset: list, csv_filename: str):
    """将最终的数据集保存为 CSV 文件。"""
    # 动态获取所有可能的键名
    all_keys = set()
    for item in memory_dataset:
        all_keys.update(item.keys())
    fieldnames = sorted(list(all_keys))

    with open(csv_filename, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in memory_dataset:
            processed_row = {}
            for key, value in row.items():
                # 将复杂的 dict/list 转换为 JSON 字符串以便在 CSV 中存储
                if isinstance(value, (dict, list)):
                    processed_row[key] = json.dumps(value, indent=2, ensure_ascii=False)
                else:
                    processed_row[key] = value
            writer.writerow(processed_row)
    print(f"✅ CSV saved to {csv_filename}")

def strip_brackets_blocks(text: str) -> Tuple[str, List[str]]:
    """
    从 assistant.value 中抽取最外层的 [ ... ] 代码块（可能有且仅有一个块）。
    返回：(去掉方括号块后的文本, [方括号块内容字符串])
    """
    if not text:
        return "", []

    # 寻找第一个 '[' 与其匹配的 ']'
    start = text.find('[')
    if start == -1:
        return text.strip(), []

    # 只处理第一个方括号块 —— 按你给的示例，assistant 工具调用就是一整个方括号块
    i = start
    depth = 0
    end = -1
    for j, ch in enumerate(text[start:], start=start):
        if ch == '[':
            depth += 1
        elif ch == ']':
            depth -= 1
            if depth == 0:
                end = j
                break

    if end == -1:
        # 方括号不配对，原样返回
        return text.strip(), []

    bracket_block = text[start + 1:end].strip()  # 去掉外层 []
    # 前后剩余文本拼起来作为 content
    remaining = (text[:start] + text[end + 1:]).strip()
    return remaining, [bracket_block]

def split_top_level_calls(block: str) -> List[str]:
    """
    将方括号内的函数调用列表按顶层逗号分割，返回每个 `Name(args)` 子串。
    例如：'A(x=1, y=2), B(z="a,b")' -> ['A(x=1, y=2)', 'B(z="a,b")']
    """
    calls = []
    buf = []
    depth = 0  # 括号深度
    in_str = False
    str_q = None

    def flush():
        s = ''.join(buf).strip()
        if s:
            calls.append(s)
        buf.clear()

    for ch in block:
        buf.append(ch)
        if in_str:
            if ch == str_q:
                in_str = False
                str_q = None
        else:
            if ch in ('"', "'"):
                in_str = True
                str_q = ch
            elif ch == '(':
                depth += 1
            elif ch == ')':
                depth = max(0, depth - 1)
            elif ch == ',' and depth == 0:
                # 顶层分割
                buf.pop()  # 去掉这个逗号
                flush()
    flush()
    # 过滤空串
    return [c for c in calls if c]

def parse_call(call_str: str) -> Tuple[str, str]:
    """
    解析单个调用：'Function Name(arg1=..., ...)'
    返回：(function_name, arguments_json_like_string)
    不强行解析参数为 dict，只保留原始参数字符串（与示例一致：arguments 是字符串）。
    """
    # 匹配 'Name(' 开头，名称允许包含空格等（如 'Market Trends API'）
    m = re.match(r'\s*(.+?)\s*\(', call_str)
    if not m:
        # 没有参数的小函数，或格式异常；退化处理为全名、空参数
        return call_str.strip(), "{}"
    name = m.group(1).strip()
    # 提取最后一个右括号对应的参数体
    # 简单法：找到第一个 '(' 与其配对的 ')'
    start = call_str.find('(')
    depth = 0
    end = -1
    for i, ch in enumerate(call_str[start:], start=start):
        if ch == '(':
            depth += 1
        elif ch == ')':
            depth -= 1
            if depth == 0:
                end = i
                break
    args = call_str[start + 1:end].strip() if end != -1 else ""
    # 如果没有任何内容，统一输出 "{}"；否则以原样字符串包起来
    arguments = args if args else ""
    # 让 arguments 符合示例：是一个 JSON 字符串（即原样字符串）
    # 这里保证是 JSON 字符串：如果本身不是以 { 开头，你给的示例也用字符串承载
    if not arguments:
        arguments_json = "{}"
    else:
        # 保持与示例一致：直接把原参数字符串作为 JSON 字符串（无需转义复杂字符）
        # 例如 'trend_type="MARKET_INDEXES", country="us"'
        arguments_json = arguments
    # 最终把 arguments 放进 JSON 字符串字段里：示例中是双引号包住的字符串
    # 为了简单，这里直接用 json.dumps 包一层，生成合法 JSON 字符串
    return name, json.dumps(arguments_json)

def parse_tool_value_to_list(tool_value: str) -> List[Dict[str, Any]]:
    """
    解析 tool 消息里的 value。通常是一个 JSON 字符串，
    形如：
      [{"name": "...","results": {...}}, ...]
    返回该列表；若解析失败，则作为一个通用 payload 放回（保持与样例一致的 content）。
    """
    try:
        data = json.loads(tool_value)
        if isinstance(data, list):
            return data
        # 某些场景也可能是单个对象
        if isinstance(data, dict):
            return [data]
        # 其他类型一律包装
        return [{"name": "tool", "results": data}]
    except Exception:
        # 不是 JSON，原样包装
        return [{"name": "tool", "results": tool_value}]

def convert_conversations(conversations: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []
    call_counter = 0

    pending_calls: List[Tuple[str, str]] = []

    for msg in conversations:
        role = msg.get("from")
        value = msg.get("value", "")

        if role == "user":
            output.append({"role": "user", "content": value})

        elif role == "assistant":
            content_text, blocks = strip_brackets_blocks(value)
            assistant_entry: Dict[str, Any] = {"role": "assistant", "content": content_text or ""}

            tool_calls_payload = []
            for block in blocks:
                calls = split_top_level_calls(block)
                for c in calls:
                    fname, fargs = parse_call(c)
                    fname = ' '.join(fname.split())
                    call_counter += 1
                    call_id = f"call_{call_counter}"
                    tool_calls_payload.append({
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": fname,
                            "arguments": fargs
                        }
                    })
                    pending_calls.append((call_id, fname))

            if tool_calls_payload:
                assistant_entry["tool_calls"] = tool_calls_payload

            output.append(assistant_entry)

        elif role == "tool":
            # 将当前 tool 的 value 拆成若干条结果，与 pending_calls 逐一匹配
            results = parse_tool_value_to_list(value)
            for res in results:
                if not pending_calls:
                    # 若没有待匹配调用，仍然输出一条匿名 tool 消息
                    output.append({
                        "role": "tool",
                        "tool_call_id": None,
                        "name": res.get("name", "tool"),
                        "content": json.dumps(res.get("results"))
                            if not isinstance(res.get("results"), str)
                            else res.get("results")
                    })
                    continue

                call_id, fname = pending_calls.pop(0)
                # content：与示例一致，直接放 JSON 字符串
                payload = res.get("results")
                content_str = payload if isinstance(payload, str) else json.dumps(payload)

                output.append({
                    "role": "tool",
                    "tool_call_id": call_id,
                    "name": res.get("name", fname),
                    "content": content_str
                })

        else:
            # 未知角色，原样作为用户文本保存
            output.append({"role": role or "user", "content": value})

    return output

def build_conversation_history(client: OpenAI, item: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Produce a simple, deterministic multi-turn conversation using the task_data and its ground_truth.
    This avoids nesting multiple LLM hops and keeps the script stable & fast.
    """
    # A minimal, readable reference conversation (adjust as needed)
    output_sample = [
        {
            "role": "user",
            "content": "Hello! Please move `final_report.pdf` into a new `temp` folder within `document`."
        },
        {
            "role": "assistant",
            "content": "Sure — creating folder and moving the file now.",
            "tool_calls": [
                {"id": "call_1", "type": "function",
                 "function": {"name": "cd", "arguments": '{"folder": "document"}'}},
                {"id": "call_2", "type": "function",
                 "function": {"name": "mkdir", "arguments": '{"dir_name": "temp"}'}},
                {"id": "call_3", "type": "function",
                 "function": {"name": "mv", "arguments": '{"source": "final_report.pdf", "destination": "temp"}'}}
            ]
        },
        {"role": "tool", "tool_call_id": "call_1", "name": "cd",
         "content": '{"status": "success", "cwd": "/workspace/document"}'},
        {"role": "tool", "tool_call_id": "call_2", "name": "mkdir",
         "content": '{"status": "success", "message": "Directory temp created."}'},
        {"role": "tool", "tool_call_id": "call_3", "name": "mv",
         "content": '{"status": "success", "message": "Moved final_report.pdf to temp."}'},

        {
            "role": "user",
            "content": "Great. Now grep the file for 'budget analysis'."
        },
        {
            "role": "assistant",
            "content": "Entering temp and running grep...",
            "tool_calls": [
                {"id": "call_4", "type": "function",
                 "function": {"name": "cd", "arguments": '{"folder": "temp"}'}},
                {"id": "call_5", "type": "function",
                 "function": {"name": "grep", "arguments": '{"file_name": "final_report.pdf", "pattern": "budget analysis"}'}}
            ]
        },
        {"role": "tool", "tool_call_id": "call_4", "name": "cd",
         "content": '{"status": "success", "cwd": "/workspace/document/temp"}'},
        {"role": "tool", "tool_call_id": "call_5", "name": "grep",
         "content": '{"status": "success", "matches": ["...budget analysis..."]}'},

        {
            "role": "user",
            "content": "Sort that file by lines, then compare with previous_report.pdf placed in temp too."
        },
        {
            "role": "assistant",
            "content": "Sorting, moving, and diffing...",
            "tool_calls": [
                {"id": "call_6", "type": "function",
                 "function": {"name": "sort", "arguments": '{"file_name": "final_report.pdf"}'}},
                {"id": "call_7", "type": "function",
                 "function": {"name": "cd", "arguments": '{"folder": ".."}'}},
                {"id": "call_8", "type": "function",
                 "function": {"name": "mv", "arguments": '{"source": "previous_report.pdf", "destination": "temp"}'}},
                {"id": "call_9", "type": "function",
                 "function": {"name": "cd", "arguments": '{"folder": "temp"}'}},
                {"id": "call_10", "type": "function",
                 "function": {"name": "diff", "arguments": '{"file_name1": "final_report.pdf", "file_name2": "previous_report.pdf"}'}}
            ]
        },
        {"role": "tool", "tool_call_id": "call_6", "name": "sort",
         "content": '{"status": "success"}'},
        {"role": "tool", "tool_call_id": "call_7", "name": "cd",
         "content": '{"status": "success", "cwd": "/workspace/document"}'},
        {"role": "tool", "tool_call_id": "call_8", "name": "mv",
         "content": '{"status": "success", "message": "Moved previous_report.pdf to temp."}'},
        {"role": "tool", "tool_call_id": "call_9", "name": "cd",
         "content": '{"status": "success", "cwd": "/workspace/document/temp"}'},
        {"role": "tool", "tool_call_id": "call_10", "name": "diff",
         "content": '{"status": "success", "diff": "..."}'}
    ]

    nature_prompt = (
    "Construct the dialogue history (Memory Dataset) : Transform the original task instructions and corresponding function calls (ground truth) into a multi-round natural dialogue that includes user questions, AI assistant responses, and Tool calls (Tool Use).\n"
    "Output Format:\n"
    f"{json.dumps(output_sample, indent=2, ensure_ascii=False)}\n"
    "Input:\n"
    f"{json.dumps(item, indent=2,ensure_ascii=False)}"
    )
    max_retries = 10
    base_delay = 1  # seconds

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="Qwen/Qwen3-Next-80B-A3B-Instruct",
                messages=[{"role": "user", "content": nature_prompt}],
                temperature=0.2,
            )
            raw = response.choices[0].message.content
            cleaned = raw.replace('```json', '').replace('```', '').strip()
            conversation_history = json.loads(cleaned)
            return conversation_history
        except Exception as e:
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                print(f"Error generating memory dataset, retry {attempt + 1}/{max_retries}: {e}, sleeping {delay}s")
                time.sleep(delay)
            else:
                print(f"Error generating memory dataset after {max_retries} attempts: {e}")
                return []

def _extract_tools_from_system_line(s: str) -> Optional[Dict[str, Any]]:
    """Attempts to extract tools from a system line, handling malformed JSON.
    Returns the extracted tools (dict) or None if parsing fails.
    """


    attempts = [s, s[:-1], s[:-2]]
    for attempt_str in attempts:
        try:
            return json.loads(attempt_str)
        except json.JSONDecodeError:
            continue # Try the next attempt
    return None # All attempts failed


# ----------------------------
# 主流程
# ----------------------------
def main():
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_BASE_URL"))
    with open(r"original_data\ToolACE\data.json", encoding="utf-8") as f:
        raw_dataset = json.load(f)
    print(f"✅ 从本地加载了 {len(raw_dataset)} 个样本。")

    jsonl_path = "toolace_formatted_conversations.jsonl"
    # 读取已处理 id
    done = {json.loads(l)["id"] for l in open(jsonl_path, encoding="utf-8")} if os.path.exists(jsonl_path) else set()

    # 预先筛选需要处理的样本，并抽取 tools，避免在线程中做 IO/过滤
    tasks = []  # (rid, conv, use_tools)
    for idx, item in enumerate(raw_dataset):
        rid = f"toolace_{idx}"
        if rid in done or not 0 <= idx < 8000:
            continue

        # 提取 tools
        use_tools = None
        for s in item["system"].splitlines():
            if '"name":' in s:
                use_tools = _extract_tools_from_system_line(s)
                break
        # if use_tools is None:
        #     continue

        conv = convert_conversations(item["conversations"])
        tasks.append((rid, conv, use_tools))

    print(f"🧵 即将并行处理 {len(tasks)} 条样本……")

    from concurrent.futures import ThreadPoolExecutor, as_completed

    with open(jsonl_path, "a", encoding="utf-8") as f_out:
        with ThreadPoolExecutor(max_workers=4) as executor:
            future_to_meta = {
                executor.submit(build_conversation_history, client, conv): (rid, use_tools)
                for rid, conv, use_tools in tasks
            }

            for future in tqdm(as_completed(future_to_meta), total=len(future_to_meta), desc="Processing"):
                rid, use_tools = future_to_meta[future]
                try:
                    history = future.result()
                except Exception as e:
                    # 某条任务失败则跳过，不中断整体
                    print(f"❌ Error processing {rid}: {e}")
                    continue

                f_out.write(
                    json.dumps(
                        {"id": rid, "conversation_history": history, "use_tools": use_tools},
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    print(f"✅ JSONL saved to {jsonl_path}")

if __name__ == "__main__":
    main()