# Mem2ACTBench 数据集核心说明

## 1. 数据集概览
**Mem2ACTBench** 是一个专注于评估大语言模型在**长上下文（Long Context）**中进行**记忆驱动工具调用**能力的基准测试。

*   **核心挑战**：工具调用的参数通常不在当前轮次，而是散落在之前的长对话历史中，需要模型具备记忆检索、信息聚合和推理能力。
*   **构建策略**：通过拼接多个原始对话片段形成长 Session（包含混合话题干扰），模拟真实的复杂对话场景。
*   **规模**：包含 **2,029** 个 Session，平均每 Session 约 **12.7** 轮对话。
*   **上下文长度**：总计 **6,569,928** tokens，平均每 Session **3238** tokens，范围 1,185 ~ 15,834 tokens。

## 2. 文件结构与用途
数据集将**对话历史**与**评测任务**分离，便于灵活使用：

| 文件名 | 类型 | 用途 |
| :--- | :--- | :--- |
| **`Mem2ACT_conversation.jsonl`** | **Session数据** | 包含完整的对话历史（User/Assistant/Tool turns）。这是模型看到的“输入上下文”。 |
| **`qa_dataset.jsonl`** | **QA数据** | 包含具体的评测样本（Query + 标准答案）。这是用于评估的“标签数据”。 |
| `*_statistics.json` | 统计信息 | 数据集的统计分布概览。 |

## 3. 核心数据字段详解

### (1) Session 数据 (`Mem2ACT_conversation.jsonl`)
*   **`session_id`**: 全局唯一标识。
*   **`turns`**: 完整的对话轮次列表。
*   **`original_conversation_ids`**: 组成该长 Session 的原始对话来源（用于追踪话题混合情况）。
*   **`token_count`**: 该 Session 的总 token 数（使用 cl100k_base tokenizer 计算）。

### (2) QA 数据 (`qa_dataset.jsonl`)
这是评测的核心，每条记录包含详细的元数据：
*   **`query`**: 用户的查询文本。
*   **`tool_call`**: 期望的工具调用结果。
    *   含 **`grounding_info`**：标注每个参数的来源类型（是直接提取还是推理得到）。
*   **`complexity_metadata`**: **[重点]** 定义了该样本的难度分级。

## 4. 评估维度与难度分级 (Complexity Levels)
数据集根据**参数获取的难易程度**，定义了四个核心认知层级：

| 层级 | 名称 | 描述 |
| :--- | :--- | :--- |
| **L1** | **Direct-Copy** | 单一来源，参数直接提取。 |
| **L2** | **Semantic-Understanding** | 单一来源，需要推理或转换。 |
| **L3** | **Cross-turn-Aggregation** | 多个来源，聚合信息且无时序冲突。 |
| **L4** | **Conflict-Resolution / Multi-hop-Reasoning** | 需要时序冲突消解，或跨来源推理。 |

此外，还细分了参数类型：
*   **`explicit`**: 显式提及。
*   **`inferred`**: 隐式推断。
*   **`default`**: 使用工具默认值。

## 5. 快速使用逻辑
1.  加载 `qa_dataset.jsonl` 获取评测问题和标准答案。
2.  根据 `qa_dataset` 中的 `source_conversation_ids`，从 `Mem2ACT_conversation.jsonl` 中检索对应的完整对话历史作为 Context。
3.  将 Context + Query 输入模型，对比模型输出与 `tool_call` 标签。

***

**总结**：Mem2ACTBench 不仅仅是做工具调用，更是在考察模型**“在长对话中记得住、找得准、推得对”**的能力。通过 L1-L4 的分级，可以精细化诊断模型在记忆检索和逻辑推理上的短板。
