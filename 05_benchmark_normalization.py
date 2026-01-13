"""
Benchmark数据集规范化脚本
将conversation_sequence.csv和memory_driven_qa.jsonl规范化为标准格式
"""

import json
import csv
import pandas as pd
import random
from typing import List, Dict, Any, Set
from pathlib import Path
from collections import defaultdict
import tiktoken


class BenchmarkNormalizer:
    """Benchmark数据集规范化器"""
    
    def __init__(self, conversation_csv: str, qa_jsonl: str, output_dir: str):
        self.conversation_csv = Path(conversation_csv)
        self.qa_jsonl = Path(qa_jsonl)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # 初始化tokenizer (使用cl100k_base，适用于GPT-4等模型)
        try:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        except Exception as e:
            print(f"⚠️  Tokenizer初始化失败: {e}，将使用近似估算")
            self.tokenizer = None
        
    def load_qa_data(self) -> Dict[str, List[Dict]]:
        """加载QA数据并按thread_id分组"""
        print("📖 加载QA数据...")
        qa_by_thread = defaultdict(list)
        
        with open(self.qa_jsonl, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if line.strip():
                    try:
                        qa_item = json.loads(line)
                        thread_id = qa_item.get('thread_id', f'unknown_{line_num}')
                        qa_by_thread[thread_id].append(qa_item)
                    except json.JSONDecodeError as e:
                        print(f"⚠️  第{line_num}行JSON解析错误: {e}")
        
        print(f"✅ 加载完成: {len(qa_by_thread)}个thread, 共{sum(len(v) for v in qa_by_thread.values())}条QA")
        return dict(qa_by_thread)
    
    def extract_source_id_from_conversation(self, conv_id: str) -> str:
        """从conversation_id提取source_id前缀"""
        # toolace_3552 -> toolace
        # oss_7563 -> oss
        # multi_turn_base_51 -> multi_turn_base
        parts = conv_id.split('_')
        if len(parts) >= 2:
            return '_'.join(parts[:-1])
        return conv_id
    
    def process_conversations(self, qa_by_thread: Dict[str, List[Dict]], target_turns: int = 12) -> List[Dict]:
        """处理对话数据,合并为长sessions"""
        print("\n📊 处理对话数据 (合并模式, target_turns={})...".format(target_turns))
        
        sessions = []
        session_id = 0
        
        # 累积缓冲区
        current_turns = []
        current_conv_ids = []
        
        # 使用csv.reader处理大文件
        with open(self.conversation_csv, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            for row_num, row in enumerate(reader, 1):
                conv_id = row['id']
                try:
                    conversation_history = json.loads(row['conversation_history'])
                except json.JSONDecodeError as e:
                    print(f"⚠️  {conv_id}: conversation_history解析错误")
                    continue
                
                # 累积数据
                # 为每个turn添加source_id标记
                for turn in conversation_history:
                    turn['source_id'] = conv_id
                
                current_turns.extend(conversation_history)
                current_conv_ids.append(conv_id)
                
                # 检查是否达到目标长度
                user_turns_count = sum(1 for t in current_turns if t['role'] == 'user')
                
                if user_turns_count >= target_turns:
                    session_id += 1
                    
                    session_data = {
                        'session_id': f'session_{session_id:05d}',
                        'original_conversation_ids': current_conv_ids,
                        'turns': current_turns,
                        'turn_count': user_turns_count,
                        'has_tool_calls': any(t.get('tool_calls') for t in current_turns)
                    }
                    sessions.append(session_data)
                    
                    # 重置缓冲区
                    current_turns = []
                    current_conv_ids = []
                
                if row_num % 1000 == 0:
                    print(f"  处理进度: {row_num} conversations -> {session_id} sessions")
        
        # 处理剩余的缓冲区
        if current_turns:
            session_id += 1

            session_data = {
                'session_id': f'session_{session_id:06d}',
                'original_conversation_ids': current_conv_ids,
                'turns': current_turns,
                'turn_count': sum(1 for t in current_turns if t['role'] == 'user'),
                'has_tool_calls': any(t.get('tool_calls') for t in current_turns)
            }
            sessions.append(session_data)

        print(f"✅ 处理完成: {len(sessions)}个sessions (合并模式)")
        return sessions
    
    def split_into_sessions(self, conversation: List[Dict], max_turns: int = 10) -> List[List[Dict]]:
        """
        将长对话分割为多个session
        
        规则:
        1. 每个session最多max_turns个user轮次
        2. 保持完整的user-assistant-tool交互组
        3. 如果遇到新的独立话题,分割session
        """
        if not conversation:
            return []
        
        sessions = []
        current_session = []
        user_turn_count = 0
        
        for turn in conversation:
            current_session.append(turn)
            
            if turn['role'] == 'user':
                user_turn_count += 1
                
                # 如果达到最大轮次,分割session
                if user_turn_count >= max_turns:
                    sessions.append(current_session)
                    current_session = []
                    user_turn_count = 0
        
        # 添加最后一个session
        if current_session:
            sessions.append(current_session)
        
        return sessions if sessions else [conversation]

    def create_standard_benchmark_format(self, sessions: List[Dict]) -> Dict:
        """创建标准benchmark格式"""
        print("\n📝 生成标准benchmark格式...")
        
        benchmark = {
            'metadata': {
                'dataset_name': 'Mem2ACT-Benchmark',
                'version': '1.0',
                'total_sessions': len(sessions),
                'description': 'Memory-driven tool calling benchmark dataset'
            },
            'sessions': sessions
        }
        
        return benchmark
    
    def count_tokens_in_turn(self, turn: Dict) -> int:
        """计算单个turn的token数"""
        # 将turn转换为文本格式
        text_parts = []
        
        if 'content' in turn and turn['content']:
            text_parts.append(str(turn['content']))
        
        if 'tool_calls' in turn and turn['tool_calls']:
            text_parts.append(json.dumps(turn['tool_calls'], ensure_ascii=False))
        
        if 'tool_call_id' in turn:
            text_parts.append(str(turn.get('tool_call_id', '')))
        
        full_text = ' '.join(text_parts)
        
        if self.tokenizer:
            return len(self.tokenizer.encode(full_text))
        else:
            # 近似估算: 1 token ≈ 4 字符（中文）或 0.75 个单词（英文）
            return len(full_text) // 3
    
    def count_tokens_in_session(self, session: Dict) -> int:
        """计算session的总token数"""
        total_tokens = 0
        for turn in session.get('turns', []):
            total_tokens += self.count_tokens_in_turn(turn)
        return total_tokens
    
    def generate_statistics(self, sessions: List[Dict]) -> Dict:
        """生成统计信息"""
        print("\n📊 计算统计信息（包括token数）...")
        
        # 计算每个session的token数
        token_counts = []
        for session in sessions:
            token_count = self.count_tokens_in_session(session)
            session['token_count'] = token_count  # 保存到session中
            token_counts.append(token_count)
        
        total_tokens = sum(token_counts)
        avg_tokens = total_tokens / len(sessions) if sessions else 0
        
        stats = {
            'total_sessions': len(sessions),
            'avg_turns_per_session': sum(s['turn_count'] for s in sessions) / len(sessions) if sessions else 0,
            'sessions_with_tools': sum(1 for s in sessions if s['has_tool_calls']),
            'total_tokens': total_tokens,
            'avg_tokens_per_session': avg_tokens,
            'max_tokens_per_session': max(token_counts) if token_counts else 0,
            'min_tokens_per_session': min(token_counts) if token_counts else 0
        }
        
        print(f"✅ Token统计完成: 总计 {total_tokens:,} tokens")
        
        return stats
    
    def generate_qa_statistics(self, qa_dataset: List[Dict]) -> Dict:
        """生成QA数据集的统计信息，包括难度分级分布"""
        print("\n📊 计算QA数据集统计信息...")
        
        # 统计难度分级分布
        level_distribution = {
            'L1': 0,
            'L2': 0,
            'L3': 0,
            'L4': 0
        }
        
        param_count_distribution = defaultdict(int)
        
        for qa in qa_dataset:
            complexity = qa.get('complexity_metadata', {})
            level = complexity.get('level', 'unknown')
            if level in level_distribution:
                level_distribution[level] += 1
            
            param_count = complexity.get('param_count', 0)
            param_count_distribution[param_count] += 1
        
        stats = {
            'total_qa': len(qa_dataset),
            'level_distribution': level_distribution,
            'param_count_distribution': dict(param_count_distribution)
        }
        
        print(f"✅ QA统计完成:")
        print(f"  - 总QA数: {len(qa_dataset)}")
        print(f"  - 难度分布: L1={level_distribution['L1']}, L2={level_distribution['L2']}, L3={level_distribution['L3']}, L4={level_distribution['L4']}")
        
        return stats
    
    def recalculate_complexity_level(self, tool_call: Dict, evolution_chain: List[Dict] = None) -> Dict:
        """
        基于认知负荷的难度分级
        
        L1: 直接复制粘贴 - 信息在单一来源，字面匹配，全是explicit
        L2: 需要语义理解 - 单一来源，需要推理/转换，有inferred
        L3: 需要跨轮次聚合 - 多个来源，信息分散但无冲突，全是explicit
        L4: 需要推理或冲突消解 - 多个来源且有inferred，或有时序冲突
        """
        grounding_info = tool_call.get('grounding_info', {})
        arguments = tool_call.get('arguments', {})
        relevant_source_ids = tool_call.get('relevant_source_ids', [])
        
        # 统计参数类型（排除default）
        explicit_count = 0
        inferred_count = 0
        
        for param_name, param_value in arguments.items():
            if param_name in grounding_info:
                param_info = grounding_info[param_name]
                param_type = param_info.get('type', 'unknown')
                
                if param_type == 'explicit':
                    explicit_count += 1
                elif param_type == 'inferred':
                    inferred_count += 1
        
        # 有效参数数量（排除default）
        effective_param_count = explicit_count + inferred_count
        
        # 来源数量
        source_count = len(relevant_source_ids) if relevant_source_ids else 1
        
        # 检查是否有时序冲突（通过evolution_chain判断）
        has_temporal_conflict = False
        if evolution_chain:
            # 如果同一个attribute有多个不同的source_id，说明有时序更新
            attribute_sources = {}
            for item in evolution_chain:
                attr = item.get('attribute', '')
                source_id = item.get('source_id', '')
                if attr:
                    if attr not in attribute_sources:
                        attribute_sources[attr] = []
                    attribute_sources[attr].append(source_id)
            
            # 检查是否有属性来自多个不同的source
            has_temporal_conflict = any(len(set(sources)) > 1 for sources in attribute_sources.values())
        
        # 难度分级逻辑
        if source_count == 1:
            # 单一来源
            if inferred_count == 0:
                # 全是explicit，直接复制粘贴
                level = 'L1'
                level_name = 'Direct-Copy'
                reasoning = '信息在单一来源，字面匹配'
            else:
                # 有inferred，需要语义理解
                level = 'L2'
                level_name = 'Semantic-Understanding'
                reasoning = '单一来源，需要推理或转换'
        else:
            # 多个来源
            if has_temporal_conflict:
                # 有时序冲突，最高难度
                level = 'L4'
                level_name = 'Conflict-Resolution'
                reasoning = '存在时序冲突，需要消解矛盾'
            elif inferred_count > 0:
                # 多来源 + 需要推理
                level = 'L4'
                level_name = 'Multi-hop-Reasoning'
                reasoning = '多个来源且需要推理'
            else:
                # 多来源但全是explicit，只需聚合
                level = 'L3'
                level_name = 'Cross-turn-Aggregation'
                reasoning = '信息分散在多个来源，需要聚合'
        
        return {
            'level': level,
            'level_name': level_name,
            'reasoning': reasoning,
            'source_count': source_count,
            'explicit_count': explicit_count,
            'inferred_count': inferred_count,
            'effective_param_count': effective_param_count,
            'has_temporal_conflict': has_temporal_conflict
        }
    
    def extract_qa_dataset(self, qa_by_thread: Dict[str, List[Dict]]) -> List[Dict]:
        """从原始QA数据提取所有QA，包含完整的标签和元数据"""
        print("\n📦 提取QA数据集...")
        
        qa_dataset = []
        qa_id = 0
        
        for thread_id, qa_items in qa_by_thread.items():
            for qa_item in qa_items:
                qa_pair = qa_item.get('qa_pair', {})
                tool_calls = qa_pair.get('tool_calls', [])
                source_records = qa_item.get('source_records', [])
                evolution_chain = qa_item.get('evolution_chain', [])
                tool_call = tool_calls[0]
                qa_id += 1
                
                # 重新计算难度分级（传入evolution_chain用于检测时序冲突）
                complexity_metadata = self.recalculate_complexity_level(tool_call, evolution_chain)
                
                qa_record = {
                    'qa_id': f'qa_{qa_id:03d}',
                    'source_conversation_ids': source_records,
                    'evolution_chain': evolution_chain,
                    'query': qa_pair.get('query', ''),
                    'tool_call': {
                        'name': tool_call.get('name', ''),
                        'arguments': tool_call.get('arguments', {}),
                        'grounding_info': tool_call.get('grounding_info', {})
                    },
                    'target_tool_schema':qa_item.get('target_tool_schema', {}),
                    'complexity_metadata': complexity_metadata,
                }
                
                qa_dataset.append(qa_record)
    
        print(f"✅ 提取完成: {len(qa_dataset)}条QA记录")
        return qa_dataset
    
    def save_qa_dataset(self, qa_dataset: List[Dict]):
        """保存QA数据集到单独文件"""
        print("\n💾 保存QA数据集...")
        
        # 保存为JSONL格式
        qa_output = self.output_dir / 'qa_dataset.jsonl'
        with open(qa_output, 'w', encoding='utf-8') as f:
            for qa in qa_dataset:
                f.write(json.dumps(qa, ensure_ascii=False) + '\n')
        print(f"✅ 保存QA数据集: {qa_output}")
        
    
    def save_normalized_data(self, benchmark: Dict, stats: Dict):
        """保存规范化数据"""
        print("\n💾 保存规范化数据...")
        
        # 1. 保存完整benchmark (JSONL格式,便于大文件处理)
        # 删除thread_id, constituent_threads, qa_items字段
        output_jsonl = self.output_dir / 'Mem2ACT_conversation.jsonl'
        with open(output_jsonl, 'w', encoding='utf-8') as f:
            for session in benchmark['sessions']:
                # 创建清理后的session副本
                cleaned_session = {
                    'session_id': session['session_id'],
                    'original_conversation_ids': session['original_conversation_ids'],
                    'turns': session['turns'],
                    'turn_count': session['turn_count'],
                    'has_tool_calls': session['has_tool_calls'],
                    'token_count': session.get('token_count', 0)
                }
                f.write(json.dumps(cleaned_session, ensure_ascii=False) + '\n')
        print(f"✅ 保存benchmark: {output_jsonl}")
    
        # 3. 保存统计信息
        stats_file = self.output_dir / 'benchmark_statistics.json'
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        print(f"✅ 保存统计: {stats_file}")
    
        # 6. 生成README和数据格式文档
        self._generate_readme(stats, benchmark['sessions'])
    
    def create_small_version(self, qa_dataset: List[Dict], sessions: List[Dict], 
                            num_qa: int = 350, output_suffix: str = '_small'):
        """
        创建小版本数据集
        
        Args:
            qa_dataset: 完整的QA数据集
            sessions: 完整的session列表
            num_qa: 要选取的QA数量
            output_suffix: 输出文件后缀
            
        Returns:
            (小版本QA列表, 小版本sessions列表)
        """
        print(f"\n📦 创建小版本数据集 (选取 {num_qa} 个QA)...")
        
        # 1. 随机采样QA（设置seed保证可复现）
        random.seed(42)
        sampled_qa = random.sample(qa_dataset, min(num_qa, len(qa_dataset)))
        print(f"✅ 采样完成: {len(sampled_qa)} 条QA")
        
        # 2. 收集所有相关的 conversation_ids
        required_conv_ids: Set[str] = set()
        for qa in sampled_qa:
            source_conv_ids = qa.get('source_conversation_ids', [])
            required_conv_ids.update(source_conv_ids)
        
        print(f"✅ 需要的对话ID数量: {len(required_conv_ids)}")
        
        # 3. 筛选包含这些conversation_ids的sessions，保持原始顺序
        filtered_sessions = []
        for session in sessions:
            original_conv_ids = session.get('original_conversation_ids', [])
            # 如果session中的任何对话ID在需要的集合中，则保留该session
            if any(conv_id in required_conv_ids for conv_id in original_conv_ids):
                filtered_sessions.append(session)
        
        print(f"✅ 筛选sessions: {len(filtered_sessions)} / {len(sessions)}")
        
        # 4. 重新编号QA和sessions
        for i, qa in enumerate(sampled_qa, 1):
            qa['qa_id'] = f'qa_{i:03d}'
        
        for i, session in enumerate(filtered_sessions, 1):
            session['session_id'] = f'session_{i:05d}'
        
        return sampled_qa, filtered_sessions
    
    def save_small_version(self, small_qa: List[Dict], small_sessions: List[Dict], 
                          suffix: str = '_small'):
        """保存小版本数据集"""
        print(f"\n💾 保存小版本数据集{suffix}...")
        
        # 创建小版本输出目录
        small_output_dir = self.output_dir.parent / f"{self.output_dir.name}{suffix}"
        small_output_dir.mkdir(exist_ok=True)
        
        # 1. 保存QA数据集
        qa_output = small_output_dir / 'qa_dataset.jsonl'
        with open(qa_output, 'w', encoding='utf-8') as f:
            for qa in small_qa:
                f.write(json.dumps(qa, ensure_ascii=False) + '\n')
        print(f"✅ 保存小版本QA: {qa_output} ({len(small_qa)} 条)")
        
        # 2. 保存conversations
        conv_output = small_output_dir / 'Mem2ACT_conversation.jsonl'
        with open(conv_output, 'w', encoding='utf-8') as f:
            for session in small_sessions:
                cleaned_session = {
                    'session_id': session['session_id'],
                    'original_conversation_ids': session['original_conversation_ids'],
                    'turns': session['turns'],
                    'turn_count': session['turn_count'],
                    'has_tool_calls': session['has_tool_calls'],
                    'token_count': session.get('token_count', 0)
                }
                f.write(json.dumps(cleaned_session, ensure_ascii=False) + '\n')
        print(f"✅ 保存小版本conversations: {conv_output} ({len(small_sessions)} sessions)")
        
        # 3. 生成统计信息
        token_counts_small = [s.get('token_count', 0) for s in small_sessions]
        total_tokens_small = sum(token_counts_small)
        
        small_stats = {
            'total_sessions': len(small_sessions),
            'total_qa': len(small_qa),
            'avg_turns_per_session': sum(s['turn_count'] for s in small_sessions) / len(small_sessions) if small_sessions else 0,
            'sessions_with_tools': sum(1 for s in small_sessions if s['has_tool_calls']),
            'total_tokens': total_tokens_small,
            'avg_tokens_per_session': total_tokens_small / len(small_sessions) if small_sessions else 0,
            'max_tokens_per_session': max(token_counts_small) if token_counts_small else 0,
            'min_tokens_per_session': min(token_counts_small) if token_counts_small else 0
        }
        
        stats_file = small_output_dir / 'benchmark_statistics.json'
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(small_stats, f, ensure_ascii=False, indent=2)
        print(f"✅ 保存小版本统计: {stats_file}")
        
        # 4. 生成README
        self._generate_readme_for_version(small_stats, small_sessions, small_output_dir, is_small=True)
        
        return small_output_dir
    
    def _generate_readme(self, stats, sessions):
        """生成README文档"""
        readme_content = f"""# Mem2ACTBench 数据集核心说明

## 1. 数据集概览
**Mem2ACTBench** 是一个专注于评估大语言模型在**长上下文（Long Context）**中进行**记忆驱动工具调用**能力的基准测试。

*   **核心挑战**：工具调用的参数通常不在当前轮次，而是散落在之前的长对话历史中，需要模型具备记忆检索、信息聚合和推理能力。
*   **构建策略**：通过拼接多个原始对话片段形成长 Session（包含混合话题干扰），模拟真实的复杂对话场景。
*   **规模**：包含 **{stats['total_sessions']:,}** 个 Session，平均每 Session 约 **{stats['avg_turns_per_session']:.1f}** 轮对话。
*   **上下文长度**：总计 **{stats.get('total_tokens', 0):,}** tokens，平均每 Session **{stats.get('avg_tokens_per_session', 0):.0f}** tokens，范围 {stats.get('min_tokens_per_session', 0):,} ~ {stats.get('max_tokens_per_session', 0):,} tokens。

## 2. 文件结构与用途
数据集将**对话历史**与**评测任务**分离，便于灵活使用：

| 文件名 | 类型 | 用途 |
| :--- | :--- | :--- |
| **`benchmark_normalized.jsonl`** | **Session数据** | 包含完整的对话历史（User/Assistant/Tool turns）。这是模型看到的“输入上下文”。 |
| **`qa_dataset.jsonl`** | **QA数据** | 包含具体的评测样本（Query + 标准答案）。这是用于评估的“标签数据”。 |
| `*_statistics.json` | 统计信息 | 数据集的统计分布概览。 |

## 3. 核心数据字段详解

### (1) Session 数据 (`benchmark_normalized.jsonl`)
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
数据集根据**参数获取的难易程度**，定义了三个核心认知层级：

| 层级 | 名称 | 描述 |
| :--- | :--- | :--- |
| **L1** | **Atomic-Explicit** (显式提取) | 所有参数值都在对话历史中**明确出现**，无需推理，仅需检索。 |
| **L2** | **Atomic-Inferred** (单步推理) | 至少一个参数需要结合上下文**推断**（例如：从“明天”推断具体日期，或从用户偏好推断参数）。 |
| **L3** | **Composite/Multi-hop** (多跳推理) | 需要**聚合**多个记忆片段或进行**多步逻辑**推理才能得出参数（例如：对比多个产品后选择）。 |

此外，还细分了参数类型：
*   **`explicit`**: 显式提及。
*   **`inferred`**: 隐式推断。
*   **`default`**: 使用工具默认值。

## 5. 快速使用逻辑
1.  加载 `qa_dataset.jsonl` 获取评测问题和标准答案。
2.  根据 `qa_dataset` 中的 `source_conversation_ids` 或 `session_id`，从 `benchmark_normalized.jsonl` 中检索对应的完整对话历史作为 Context。
3.  将 Context + Query 输入模型，对比模型输出与 `tool_call` 标签。

***

**总结**：Mem2ACTBench 不仅仅是做工具调用，更是在考察模型**“在长对话中记得住、找得准、推得对”**的能力。通过 L1-L3 的分级，可以精细化诊断模型在记忆检索和逻辑推理上的短板。
"""
        
        readme_path = self.output_dir / 'README.md'
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(readme_content)
        print(f"✅ 保存README: {readme_path}")
    
    def _generate_readme_for_version(self, stats, sessions, output_dir: Path, is_small: bool = False):
        """生成特定版本的README文档"""
        version_note = "**小版本数据集**，用于快速开发和测试。" if is_small else "完整版数据集。"
        
        readme_content = f"""# Mem2ACTBench 数据集{'（小版本）' if is_small else ''}

## 1. 数据集概览
**Mem2ACTBench** 是一个专注于评估大语言模型在**长上下文（Long Context）**中进行**记忆驱动工具调用**能力的基准测试。

{version_note}

*   **核心挑战**：工具调用的参数通常不在当前轮次，而是散落在之前的长对话历史中，需要模型具备记忆检索、信息聚合和推理能力。
*   **构建策略**：通过拼接多个原始对话片段形成长 Session（包含混合话题干扰），模拟真实的复杂对话场景。
*   **规模**：包含 **{stats['total_sessions']:,}** 个 Session，平均每 Session 约 **{stats['avg_turns_per_session']:.1f}** 轮对话{f"，共 **{stats.get('total_qa', 0)}** 条QA" if 'total_qa' in stats else ''}。
*   **上下文长度**：总计 **{stats.get('total_tokens', 0):,}** tokens，平均每 Session **{stats.get('avg_tokens_per_session', 0):.0f}** tokens，范围 {stats.get('min_tokens_per_session', 0):,} ~ {stats.get('max_tokens_per_session', 0):,} tokens。

## 2. 文件结构与用途
数据集将**对话历史**与**评测任务**分离，便于灵活使用：

| 文件名 | 类型 | 用途 |
| :--- | :--- | :--- |
| **`Mem2ACT_conversation.jsonl`** | **Session数据** | 包含完整的对话历史（User/Assistant/Tool turns）。这是模型看到的"输入上下文"。 |
| **`qa_dataset.jsonl`** | **QA数据** | 包含具体的评测样本（Query + 标准答案）。这是用于评估的"标签数据"。 |
| `benchmark_statistics.json` | 统计信息 | 数据集的统计分布概览。 |

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
数据集根据**参数获取的难易程度**，定义了三个核心认知层级：

| 层级 | 名称 | 描述 |
| :--- | :--- | :--- |
| **L1** | **Atomic-Explicit** (显式提取) | 所有参数值都在对话历史中**明确出现**，无需推理，仅需检索。 |
| **L2** | **Atomic-Inferred** (单步推理) | 至少一个参数需要结合上下文**推断**（例如：从"明天"推断具体日期，或从用户偏好推断参数）。 |
| **L3** | **Composite/Multi-hop** (多跳推理) | 需要**聚合**多个记忆片段或进行**多步逻辑**推理才能得出参数（例如：对比多个产品后选择）。 |

此外，还细分了参数类型：
*   **`explicit`**: 显式提及。
*   **`inferred`**: 隐式推断。
*   **`default`**: 使用工具默认值。

## 5. 快速使用逻辑
1.  加载 `qa_dataset.jsonl` 获取评测问题和标准答案。
2.  根据 `qa_dataset` 中的 `source_conversation_ids`，从 `Mem2ACT_conversation.jsonl` 中检索对应的完整对话历史作为 Context。
3.  将 Context + Query 输入模型，对比模型输出与 `tool_call` 标签。

***

**总结**：Mem2ACTBench 不仅仅是做工具调用，更是在考察模型**"在长对话中记得住、找得准、推得对"**的能力。通过 L1-L3 的分级，可以精细化诊断模型在记忆检索和逻辑推理上的短板。
"""
        
        readme_path = output_dir / 'README.md'
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(readme_content)
        print(f"✅ 保存README: {readme_path}")

    
    def run(self, create_small: bool = True, small_qa_count: int = 400):
        """执行规范化流程"""
        print("🚀 开始Benchmark数据集规范化\n")
        print("=" * 60)
        
        # 1. 加载QA数据
        qa_by_thread = self.load_qa_data()
        
        # 2. 提取QA数据集（直接从原始QA数据，包含重新计算的难度分级）
        qa_dataset = self.extract_qa_dataset(qa_by_thread)
        
        # 3. 生成QA统计信息
        qa_stats = self.generate_qa_statistics(qa_dataset)
        
        # 4. 保存QA数据集
        self.save_qa_dataset(qa_dataset)
        
        # 5. 处理对话数据（不关联QA）
        sessions = self.process_conversations(qa_by_thread)
        
        # 6. 创建标准格式
        benchmark = self.create_standard_benchmark_format(sessions)
        
        # 7. 生成统计
        stats = self.generate_statistics(sessions)
        
        # 8. 保存数据
        self.save_normalized_data(benchmark, stats)
        
        print("\n" + "=" * 60)
        print("✅ 完整版规范化完成!")
        print(f"\n📊 完整版统计摘要:")
        print(f"  - 总Sessions: {stats['total_sessions']}")
        print(f"  - 平均每Session轮次: {stats['avg_turns_per_session']:.2f}")
        print(f"  - 包含工具调用的Sessions: {stats['sessions_with_tools']}")
        print(f"  - 总QA数量: {qa_stats['total_qa']}")
        print(f"  - 难度分布: L1={qa_stats['level_distribution']['L1']}, L2={qa_stats['level_distribution']['L2']}, L3={qa_stats['level_distribution']['L3']}, L4={qa_stats['level_distribution']['L4']}")
        print(f"  - 总Token数: {stats['total_tokens']:,}")
        print(f"  - 平均每Session Token数: {stats['avg_tokens_per_session']:.0f}")
        print(f"  - Token范围: {stats['min_tokens_per_session']:,} ~ {stats['max_tokens_per_session']:,}")
        
        # 9. 创建小版本数据集
        if create_small:
            print("\n" + "=" * 60)
            print(f"🔄 开始创建小版本数据集...")
            small_qa, small_sessions = self.create_small_version(
                qa_dataset, sessions, num_qa=small_qa_count
            )
            
            # 10. 生成小版本QA统计
            small_qa_stats = self.generate_qa_statistics(small_qa)
            
            # 11. 保存小版本
            small_output_dir = self.save_small_version(small_qa, small_sessions)
            
            print("\n" + "=" * 60)
            print("✅ 小版本创建完成!")
            print(f"\n📊 小版本统计摘要:")
            print(f"  - 总QA数量: {small_qa_stats['total_qa']}")
            print(f"  - 难度分布: L1={small_qa_stats['level_distribution']['L1']}, L2={small_qa_stats['level_distribution']['L2']}, L3={small_qa_stats['level_distribution']['L3']}, L4={small_qa_stats['level_distribution']['L4']}")
            print(f"  - 总Sessions: {len(small_sessions)}")
            print(f"  - 平均每Session轮次: {sum(s['turn_count'] for s in small_sessions) / len(small_sessions):.2f}")
            
            # 获取小版本的token统计
            small_token_counts = [s.get('token_count', 0) for s in small_sessions]
            small_total_tokens = sum(small_token_counts)
            print(f"  - 总Token数: {small_total_tokens:,}")
            print(f"  - 平均每Session Token数: {small_total_tokens / len(small_sessions) if small_sessions else 0:.0f}")
            print(f"  - Token范围: {min(small_token_counts) if small_token_counts else 0:,} ~ {max(small_token_counts) if small_token_counts else 0:,}")
            print(f"  - 输出目录: {small_output_dir}")
        
        print("\n" + "=" * 60)
        print("🎉 所有数据集生成完成!")


def main():
    """主函数"""
    # 配置路径
    conversation_csv = "processed_data/conversation_sequence.csv"
    qa_jsonl = "processed_data/memory_driven_qa_Kimi-K2-Instruct-0905_v2.jsonl"
    output_dir = "Mem2ACTBench"
    
    # 执行规范化
    normalizer = BenchmarkNormalizer(conversation_csv, qa_jsonl, output_dir)
    normalizer.run(create_small=True, small_qa_count=400)


if __name__ == "__main__":
    main()
