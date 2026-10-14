import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="从格式化 ToolACE 对话提取去重工具池")
    parser.add_argument("-i", "--input", type=Path,
                        default=Path("toolace_formatted_conversations.jsonl"),
                        help="格式化对话 JSONL 路径")
    parser.add_argument("-o", "--output", type=Path, default=Path("tool_api_pool.json"),
                        help="工具池 JSON 路径（覆盖写入；新工具池应配套重建嵌入缓存）")
    args = parser.parse_args()

    tool_pool = set()
    with args.input.open("r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            for tool in data.get("use_tools", []):
                tool_pool.add(json.dumps(tool, sort_keys=True))

    unique_tools = [json.loads(tool) for tool in tool_pool]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(unique_tools, f, indent=4, ensure_ascii=False)
    print(f"Extracted {len(unique_tools)} unique tools to {args.output}")


if __name__ == "__main__":
    main()
