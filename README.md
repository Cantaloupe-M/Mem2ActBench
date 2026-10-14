# Mem2ActBench (ACL 2026)

ACL 2026（Long Papers）论文 **Mem2ActBench: A Benchmark for Evaluating Long-Term Memory Utilization in Task-Oriented Autonomous Agents** 的官方实现。

**作者：** Yiting Shen, Kun Li, Wei Zhou, Songlin Hu

[论文](Mem2Act_paper.pdf) · [官方仓库](https://github.com/Cantaloupe-M/Mem2ActBench)

Mem2ActBench 评估智能体利用长期记忆选择工具并补全调用参数的能力。本目录提供基于 ToolACE、BFCL 和 OASST1 的数据集构建流水线，包括对话构建、事实提取与冲突解析、对话排序、QA 生成及 Benchmark 规范化。

## 环境配置

Python 3.10+。在项目根目录安装依赖并配置兼容 OpenAI 的 API 服务：

```bash
pip install -r requirements.txt
export OPENAI_API_KEY="your-api-key"
export OPENAI_BASE_URL="https://api.siliconflow.cn/v1"
```

事实提取和 QA 构造使用 `BAAI/bge-m3` 嵌入模型，建议使用 GPU。

## 数据准备

将原始数据放入以下路径：

```text
original_data/
├── Berkeley-Function-Calling-Leaderboard/
├── ToolACE/data.json
└── oasst1/2023-04-12_oasst_ready.messages.jsonl
```

## 运行命令

### 1. 对话构建

```bash
python 01_conversation_builder.py bfcl \
    --data-path original_data/Berkeley-Function-Calling-Leaderboard \
    --output bfcl_formatted_conversations.jsonl

python 01_conversation_builder.py oasst1 \
    --data-path original_data/oasst1/2023-04-12_oasst_ready.messages.jsonl \
    --output oasst1_formatted_conversations.jsonl

python 01_conversation_builder.py toolace \
    --data-path original_data/ToolACE/data.json \
    --output toolace_formatted_conversations.jsonl \
    --start-idx 0 --end-idx 8000
```

### 2. 事实提取与冲突解析

```bash
python 02_extract_facts_conflicts.py \
    --input-files toolace_formatted_conversations.jsonl oasst1_formatted_conversations.jsonl bfcl_formatted_conversations.jsonl \
    --output-dir processed_data \
    --api-base-url "$OPENAI_BASE_URL" --max-workers 6
```

### 3. 对话排序

```bash
python 03_conv_sort.py \
    --evolutions processed_data/conflict_evolutions.csv \
    --facts processed_data/original_records_with_facts.csv \
    --output processed_data/conversation_sequence.csv
```

### 4. 工具池提取与 QA 构造

```bash
python extract_tool_pool.py \
    --input toolace_formatted_conversations.jsonl \
    --output tool_api_pool.json

python 04_qa_construction.py \
    --memory-path processed_data/conflict_evolutions.csv \
    --tool-pool-path tool_api_pool.json \
    --llm-model moonshotai/Kimi-K2-Instruct-0905 \
    --output processed_data/memory_driven_qa_Kimi-K2-Instruct-0905_v2.jsonl \
    --max-workers 4
```

### 5. Benchmark 规范化

```bash
python 05_benchmark_normalization.py \
    --conversation-csv processed_data/conversation_sequence.csv \
    --qa-jsonl processed_data/memory_driven_qa_Kimi-K2-Instruct-0905_v2.jsonl \
    --output-dir Mem2ACTBench --small-qa-count 400
```

输出至 `Mem2ACTBench/` 和 `Mem2ACTBench_small/`，各包含对话数据 `Mem2ACT_conversation.jsonl`、评测数据 `qa_dataset.jsonl`、统计信息及数据集说明。
各脚本的完整参数可通过 `python <script>.py --help` 查看。

## License

MIT License
