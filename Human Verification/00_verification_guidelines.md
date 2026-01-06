# ToolmemBench Human Verification Guidelines

本指南用于指导人工标注者对 ToolmemBench 数据集样本进行质量评估。我们的目标是验证数据集是否符合“基于记忆的工具调用”的设计目标。

## 评估维度 (Evaluation Dimensions)

对于每个样本，你需要从以下四个维度进行评分（1-5分）或二值判断（Yes/No）。

### 1. Tool Selection Correctness (工具选择正确性)
- **问题**: 考虑到 Memory Context 和 User Query，选择的工具是否是最合理的？
- **评分标准**:
    - **Yes (1)**: 工具完全匹配意图，且是唯一的合理选择。
    - **No (0)**: 工具选择错误，或者有更合适的工具未被选择。

### 2. Memory Dependency (记忆依赖性)
- **问题**: 如果不看 Memory Context，仅凭 User Query 是否能确定工具的所有参数？(即 Query 是否信息不全？)
- **评分标准**:
    - **Yes (1)**: Query 包含模糊指代（如 "that project", "same budget"），必须查阅 Memory 才能填全参数。这是我们想要的高质量样本。
    - **No (0)**: Query 已经包含了所有必要信息（信息泄露），不需要 Memory 即可解决。这属于低质量样本。

### 3. Argument Accuracy & Grounding (参数准确性与来源)
- **问题**: 工具调用的参数值是否正确来源于 Memory？
- **评分标准 (1-5)**:
    - **5 (Perfect)**: 所有参数都准确无误地从 Memory 中提取或推理得出。
    - **3 (Acceptable)**: 参数基本正确，但在非关键参数上有微小偏差，或者推理略显牵强但可接受。
    - **1 (Wrong)**: 参数值主要甚至是完全错误的，或者出现了 Memory 中不存在的幻觉 (Hallucination)。

### 4. Query Naturalness (查询自然度)
- **问题**: User Query 读起来是否像真实的人类语言？
- **评分标准 (1-5)**:
    - **5 (Natural)**: 非常自然，符合日常口语或工作场景的表达习惯。
    - **3 (Okay)**: 语法正确但略显生硬，或者是明显的机器翻译痕迹。
    - **1 (Unnatural)**: 语法错误，或者逻辑不通，甚至不知所云。

## 标注流程

1. **阅读 Memory Context**: 理解用户之前的交互历史和事实。
2. **阅读 User Query**: 看用户现在的请求。尝试先不看 Tool Call，自己想一下应该调用什么工具，参数是什么。
3. **检查 Tool Call**: 对比给出的 Tool Call 和你的预期。
4. **填写评分**: 在 Excel/CSV 表格中对应列填写分数。

## 注意事项

- 某些参数可能是默认值 (Default)，这些不需要在 Memory 中找到对应，只要符合工具定义即可。
- 推理型 (Inferred) 参数可能不直接出现在文本中，需要逻辑推断（例如：Memory 说“去法国”，参数是 `country="FR"`）。这也算作正确。
