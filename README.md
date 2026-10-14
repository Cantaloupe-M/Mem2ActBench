# Mem2ACTBench - 记忆驱动工具调用基准测试数据集构建工具

## 项目简介

Mem2ACTBench 是一个用于构建**记忆驱动工具调用（Memory-Driven Tool Calling）**基准测试数据集的完整流水线。该项目旨在评估大语言模型在长上下文对话中，从历史记忆中提取信息并正确调用工具的能力。

## 环境依赖

```bash
pip install -r requirements.txt
```

## Pipeline

```
原始数据 → 01_对话处理 → 02_事实提取 → 03_序列排序 → 04_QA构造 → 05_规范化 → Benchmark
```

## 快速开始

**配置环境变量**
```bash
export OPENAI_API_KEY="your-api-key"
export OPENAI_BASE_URL="https://api.siliconflow.cn/v1"  # 可选
```

### 1. `01_conversation_builder.py` - 对话数据构建器

将多个原始数据集转换为统一的对话格式。

**支持的数据集：**
- `bfcl` - Berkeley Function Calling Leaderboard
- `oasst1` - Open Assistant 数据集
- `toolace` - ToolACE 数据集

**使用方法：**
```bash
# 处理 BFCL 数据集
python 01_conversation_builder.py bfcl \
    --data-path original_data/Berkeley-Function-Calling-Leaderboard \
    --output bfcl_formatted_conversations.jsonl

# 处理 OASST1 数据集
python 01_conversation_builder.py oasst1 \
    --data-path original_data/oasst1/2023-04-12_oasst_ready.messages.jsonl \
    --output oasst1_formatted_conversations.jsonl

# 处理 ToolACE 数据集
python 01_conversation_builder.py toolace \
    --data-path original_data/ToolACE/data.json \
    --output toolace_formatted_conversations.jsonl \
    --start-idx 0 --end-idx 8000
```

**通用参数：**
| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--api-base` | OpenAI API 基础 URL | 环境变量 `OPENAI_BASE_URL` |
| `--api-key` | API 密钥 | 环境变量 `OPENAI_API_KEY` |
| `--model` | 使用的模型 | `Qwen/Qwen3-Next-80B-A3B-Instruct` |
| `--workers` | 并行线程数 | 4 |
| `-o, --output` | 输出文件路径 | 必填 |

---

### 2. `02_extract_facts_conflicts.py` - 事实提取与冲突检测

从对话中提取结构化事实，并使用 BERTopic 进行语义聚类，检测记忆演化和冲突。

**使用方法：**
```bash
python 02_extract_facts_conflicts.py \
    --input-files toolace_formatted_conversations.jsonl oasst1_formatted_conversations.jsonl \
    --output-dir processed_data \
    --model Qwen/Qwen3-Next-80B-A3B-Instruct \
    --embedding-model BAAI/bge-m3 \
    --max-workers 6
```

**参数说明：**
| 参数 | 说明 | 默认值 |
|------|------|--------|
| `-i, --input-files` | 输入 JSONL 文件列表 | 三个默认数据集 |
| `-o, --output-dir` | 输出目录 | `processed_data` |
| `-m, --model` | LLM 模型名称 | `Qwen/Qwen3-Next-80B-A3B-Instruct` |
| `--embedding-model` | 嵌入模型 | `BAAI/bge-m3` |
| `--min-cluster-size` | 最小聚类大小 | 2 |
| `-w, --max-workers` | 并行线程数 | 6 |
| `-b, --batch-size` | 嵌入批量大小 | 64 |

**输出文件：**
- `original_records_with_facts.csv` - 提取的事实记录
- `conflict_evolutions.csv` - 记忆演化与冲突分组

---

### 3. `03_conv_sort.py` - 对话序列排序

基于拓扑排序算法，将多个对话序列合并为全局一致的时序序列。

**使用方法：**
```bash
python 03_conv_sort.py
```

**输入文件：**
- `processed_data/conflict_evolutions.csv`
- `processed_data/original_records_with_facts.csv`

**输出文件：**
- `processed_data/conversation_sequence.csv` - 排序后的对话序列

**算法特点：**
- 使用优化的拓扑排序处理序列依赖
- 启发式策略处理循环冲突（丢弃出度最高的节点）

---

### 4. `04_qa_construction.py` - QA 数据构造

基于记忆链和工具池，自动生成记忆驱动的 QA 样本。

**核心功能：**
- **混合工具检索**：Dense (BGE-M3) + Sparse (BM25) 融合
- **Schema 校验**：严格验证工具调用参数
- **记忆锚定验证**：支持精确匹配和软匹配（日期转换、Token Overlap、模糊匹配、语义相似度）
- **难度分级**：基于认知负荷的 L1-L4 分级

**难度分级标准：**
| 级别 | 名称 | 描述 |
|------|------|------|
| L1 | Direct-Copy | 单一来源，字面匹配，全是 explicit |
| L2 | Semantic-Understanding | 单一来源，需要推理/转换，有 inferred |
| L3 | Cross-turn-Aggregation | 多个来源，信息分散但无冲突 |
| L4 | Conflict-Resolution | 多个来源且有 inferred，或有时序冲突 |

---

### 5. `05_benchmark_normalization.py` - Benchmark 规范化

将处理后的数据规范化为标准 Benchmark 格式。

**使用方法：**
```bash
python 05_benchmark_normalization.py
```

**输出结构：**
```
Mem2ActBench/
├── Mem2ACT_conversation.jsonl  # Session 对话数据
├── qa_dataset.jsonl            # QA 评测数据
├── benchmark_statistics.json   # 统计信息
└── README.md                   # 数据集说明

Mem2ACTbench_small/             # 小版本（用于快速测试）
├── ...
```

**功能特点：**
- 自动合并短对话为长 Session（默认 12 轮）
- Token 统计（使用 cl100k_base tokenizer）
- 自动生成小版本数据集（默认 350 条 QA）

---

### 6. `extract_tool_pool.py` - 工具池提取

从 ToolACE 数据集中提取唯一的工具定义。

**使用方法：**
```bash
python extract_tool_pool.py
```

**输出：**
- `tool_api_pool.json` - 去重后的工具 API 定义

---

## 目录结构

```
.
├── 01_conversation_builder.py    # 对话构建
├── 02_extract_facts_conflicts.py # 事实提取与冲突检测
├── 03_conv_sort.py               # 序列排序
├── 04_qa_construction.py         # QA 构造
├── 05_benchmark_normalization.py # 规范化
├── extract_tool_pool.py          # 工具池提取
├── prompts/                      # Prompt 模板
│   ├── conversation_example.json
│   └── extraction_facts.md
├── original_data/                # 原始数据集
│   ├── Berkeley-Function-Calling-Leaderboard/
│   ├── oasst1/
│   └── ToolACE/
├── processed_data/               # 中间处理结果
├── Mem2ActBench/                 # 完整版 Benchmark
├── Mem2ACTbench_small/           # 小版本 Benchmark
└── data_analysis/                # 数据分析脚本
```


## 输出数据格式

### Session 数据 (`Mem2ACT_conversation.jsonl`)
```json
{
  "session_id": "session_00001",
  "original_conversation_ids": ["toolace_123", "oss_456"],
  "turns": [
    {"role": "user", "content": "...", "source_id": "toolace_123"},
    {"role": "assistant", "content": "...", "tool_calls": [...]},
    {"role": "tool", "tool_call_id": "call_1", "name": "...", "content": "..."}
  ],
  "turn_count": 12,
  "has_tool_calls": true,
  "token_count": 2500
}
```

### QA 数据 (`qa_dataset.jsonl`)
```json
{
  "qa_id": "qa_001",
  "source_conversation_ids": ["toolace_123"],
  "evolution_chain": [...],
  "query": "用户查询文本",
  "tool_call": {
    "name": "tool_name",
    "arguments": {"param1": "value1"},
    "grounding_info": {
      "param1": {"type": "explicit", "source_fact_id": "..."}
    }
  },
  "target_tool_schema": {...},
  "complexity_metadata": {
    "level": "L2",
    "level_name": "Semantic-Understanding",
    "reasoning": "单一来源，需要推理或转换",
    "source_count": 1,
    "explicit_count": 1,
    "inferred_count": 1
  }
}
```

## License

MIT License
