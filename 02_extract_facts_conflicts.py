import os
import json
import csv
import time
import argparse
from typing import List, Dict, Any
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
import numpy as np
from tqdm import tqdm
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
from bertopic import BERTopic
from hdbscan import HDBSCAN


# ----------------------------
# 配置类
# ----------------------------
class Config:
    """集中管理配置参数"""
    def __init__(self, args):
        self.input_files = args.input_files
        self.output_dir = args.output_dir
        self.model_name = args.model
        self.embedding_model = args.embedding_model
        self.min_cluster_size = args.min_cluster_size
        self.max_workers = args.max_workers
        self.batch_size = args.batch_size
        self.api_base_url = args.api_base_url
        self.api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
        
        # 输出文件路径
        self.facts_csv_path = os.path.join(self.output_dir, "original_records_with_facts.csv")
        self.evolutions_csv_path = os.path.join(self.output_dir, "conflict_evolutions.csv")


# ----------------------------
# 模型管理器
# ----------------------------
class ModelManager:
    """管理LLM客户端和嵌入模型"""
    def __init__(self, config: Config):
        self.config = config
        self.client = OpenAI(api_key=config.api_key, base_url=config.api_base_url)
        self.sbert_model = self._init_embedding_model()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    def _init_embedding_model(self):
        model = SentenceTransformer(self.config.embedding_model)
        if torch.cuda.is_available():
            model = model.to('cuda')
            print("使用GPU模式")
        else:
            print("未检测到GPU，使用CPU模式")
        return model
    
    def encode_texts(self, texts: List[str], show_progress: bool = True) -> np.ndarray:
        """批量编码文本为向量"""
        batch_size = min(self.config.batch_size, max(8, len(texts) // 10))
        
        if torch.cuda.is_available():
            with torch.no_grad():
                embeddings = self.sbert_model.encode(
                    texts, batch_size=batch_size, 
                    show_progress_bar=show_progress, convert_to_tensor=True
                )
                embeddings = embeddings.cpu().numpy()
                torch.cuda.empty_cache()
        else:
            embeddings = self.sbert_model.encode(texts, batch_size=128, show_progress_bar=show_progress)
            embeddings = normalize(embeddings, norm='l2')
        return embeddings


# ----------------------------
# 事实提取器
# ----------------------------
class FactExtractor:
    """处理事实提取逻辑"""
    def __init__(self, model_manager: ModelManager, config: Config):
        self.model_manager = model_manager
        self.config = config
        self._load_prompt_template()
        
    def _load_prompt_template(self):
        with open('prompts/extraction_facts.md', 'r', encoding='utf-8') as f:
            self.prompt_template = f.read()
    
    def extract_facts(self, text_content: str, max_retries: int = 10) -> List[Dict]:
        """从文本中提取事实"""
        prompt = self.prompt_template.format(text_content=text_content)
        
        for attempt in range(max_retries):
            try:
                response = self.model_manager.client.chat.completions.create(
                    model=self.config.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0
                )
                content = response.choices[0].message.content.replace("```json", "").replace("```", "")
                return json.loads(content)
            except Exception as e:
                if attempt < max_retries - 1:
                    delay = 2 ** attempt
                    print(f"事实提取尝试 {attempt+1}/{max_retries} 失败: {e}, {delay}s后重试...")
                    time.sleep(delay)
                else:
                    print(f"事实提取在 {max_retries} 次尝试后失败: {e}")
                    return []


# ----------------------------
# 语义分组器
# ----------------------------
class SemanticGrouper:
    """使用BERTopic进行语义分组"""
    def __init__(self, model_manager: ModelManager, config: Config):
        self.model_manager = model_manager
        self.config = config
        
    def group_attributes(self, facts: List[Dict]) -> Dict[str, str]:
        """对属性进行聚类分组，返回属性到规范属性的映射"""
        attrs = list({f["attribute"].strip().lower() for f in facts})
        
        if len(attrs) < self.config.min_cluster_size:
            return {a: a for a in attrs}
        
        embeddings = self.model_manager.encode_texts(attrs)
        
        hdbscan_model = HDBSCAN(
            min_cluster_size=2,
            metric='euclidean',
            cluster_selection_epsilon=0.01,
            cluster_selection_method='leaf',
            prediction_data=True
        )
        
        topic_model = BERTopic(
            embedding_model=None,
            umap_model=None,
            hdbscan_model=hdbscan_model,
            verbose=False
        )
        topics, _ = topic_model.fit_transform(attrs, embeddings)
        
        groups = {}
        topic_to_canonical = {}
        
        for attr, topic in zip(attrs, topics):
            if topic == -1:
                groups[attr] = attr
                continue
            if topic not in topic_to_canonical:
                topic_to_canonical[topic] = attr
            groups[attr] = topic_to_canonical[topic]
        
        return groups


# ----------------------------
# 记忆演变解析器
# ----------------------------
class MemoryEvolutionResolver:
    """解析记忆演变和冲突"""
    def __init__(self, model_manager: ModelManager, config: Config):
        self.model_manager = model_manager
        self.config = config
        self.grouper = SemanticGrouper(model_manager, config)
        self._load_prompt_template()
        
    def _load_prompt_template(self):
        with open('prompts/resolve_fact_group原始.md', 'r', encoding='utf-8') as f:
            self.prompt_template = f.read()
    
    def _resolve_single_group(self, facts_group: List[Dict], max_retries: int = 5) -> Dict:
        """解析单个事实组"""
        statements_by_source_id = defaultdict(list)
        for f in facts_group:
            source_id, facts = f.get("source_id"), f.get("facts")
            if source_id and facts:
                statements_by_source_id[source_id].append(facts)
        
        simplified_facts = [
            {"source_id": sid, "facts": ". ".join(facts)}
            for sid, facts in statements_by_source_id.items()
        ]
        
        for attempt in range(max_retries):
            try:
                response = self.model_manager.client.chat.completions.create(
                    model=self.config.model_name,
                    messages=[{"role": "user", "content": self.prompt_template.format(
                        data=json.dumps(simplified_facts, indent=2, ensure_ascii=False)
                    )}],
                    temperature=0.2
                )
                content = response.choices[0].message.content.replace("```json", "").replace("```", "")
                data = json.loads(content)
                
                all_group_ids = set(statements_by_source_id.keys())
                expected = all_group_ids - set(data["discarded_source_ids"])
                if set(data["sorted_source_ids"]) != expected:
                    raise ValueError("sorted_source_ids 与预期不匹配")
                return data
            except Exception as e:
                if attempt < max_retries - 1:
                    delay = 2 ** attempt
                    print(f"组解析尝试 {attempt+1}/{max_retries} 失败: {e}, {delay}s后重试...")
                    time.sleep(delay)
                else:
                    print(f"组解析在 {max_retries} 次尝试后失败: {e}")
                    return None
    
    def resolve_all_groups(self, all_facts: List[Dict]) -> List[Dict]:
        """解析所有事实组"""
        if len(all_facts) < self.config.min_cluster_size:
            return []
        
        print("使用BERTopic对语义相似属性进行分组...")
        attr_map = self.grouper.group_attributes(all_facts)
        
        grouped = defaultdict(list)
        for fact in all_facts:
            original_attr = fact["attribute"].strip().lower()
            canonical_attr = attr_map.get(original_attr, original_attr)
            grouped[canonical_attr].append(fact)
        
        groups_to_process = [
            (attr, facts) for attr, facts in grouped.items()
            if len(set(f["source_id"] for f in facts)) > 1
        ]
        
        if not groups_to_process:
            print("没有找到包含多个事实的组进行解析。")
            return []
        
        print(f"找到 {len(groups_to_process)} 个组进行解析...")
        resolved_groups = []
        
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            future_to_group = {
                executor.submit(self._resolve_single_group, facts): (attr, facts)
                for attr, facts in groups_to_process
            }
            
            for future in tqdm(as_completed(future_to_group), total=len(groups_to_process), desc="解析事实组"):
                try:
                    canonical_attr, original_facts = future_to_group[future]
                    result = future.result()
                    if result:
                        resolved_groups.append({
                            "attribute_group": canonical_attr,
                            "sorted_source_ids": result.get("sorted_source_ids"),
                            "discarded_source_ids": result.get("discarded_source_ids"),
                            "narrative_summary": result.get("narrative_summary"),
                            "reasoning": result.get("reasoning"),
                            "original_facts": original_facts
                        })
                except Exception as e:
                    print(f"组解析失败: {e}")
        
        return resolved_groups


# ----------------------------
# 数据处理管道
# ----------------------------
class DataPipeline:
    """主数据处理流程"""
    def __init__(self, config: Config):
        self.config = config
        self.model_manager = ModelManager(config)
        self.extractor = FactExtractor(self.model_manager, config)
        self.resolver = MemoryEvolutionResolver(self.model_manager, config)
        
    def load_records(self) -> List[Dict]:
        """加载输入记录"""
        records = []
        for fname in self.config.input_files:
            if not os.path.exists(fname):
                print(f"警告: 文件 {fname} 不存在，跳过")
                continue
            with open(fname, encoding="utf-8") as f:
                records.extend(json.loads(line) for line in f)
        print(f"加载了 {len(records)} 条记录")
        return records
    
    def load_existing_facts(self) -> tuple:
        """加载已存在的事实数据"""
        all_facts = []
        processed_ids = set()
        
        if os.path.exists(self.config.facts_csv_path):
            print(f"从 {self.config.facts_csv_path} 加载已有事实...")
            with open(self.config.facts_csv_path, "r", encoding="utf-8-sig") as f:
                reader = csv.reader(f)
                next(reader)  # Skip header
                for row in reader:
                    record_id = row[0]
                    facts = json.loads(row[2])
                    all_facts.extend(facts)
                    processed_ids.add(record_id)
            print(f"加载了 {len(all_facts)} 条事实")
        
        return all_facts, processed_ids
    
    def extract_facts_parallel(self, records: List[Dict], processed_ids: set) -> List[Dict]:
        """并行提取事实"""
        # 准备用户文本
        user_texts = {}
        for record in records:
            if record['id'] in processed_ids:
                continue
            contents = [
                conv["content"] for conv in record.get("conversation_history", [])
                if conv.get("role") == "user" and conv.get("content")
            ]
            if contents:
                user_texts[record['id']] = ("\n".join(contents), record)
        
        if not user_texts:
            print("所有记录已处理完毕")
            return []
        
        print(f"开始并行提取 {len(user_texts)} 条记录的事实...")
        
        # 确保CSV文件存在
        os.makedirs(self.config.output_dir, exist_ok=True)
        if not os.path.exists(self.config.facts_csv_path):
            with open(self.config.facts_csv_path, "w", encoding="utf-8-sig", newline="") as f:
                csv.writer(f).writerow(["record_id", "conversation_history", "extracted_facts"])
        
        new_facts = []
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            future_to_id = {
                executor.submit(self.extractor.extract_facts, text): (rid, record)
                for rid, (text, record) in user_texts.items()
            }
            
            for future in tqdm(as_completed(future_to_id), total=len(user_texts), desc="提取事实"):
                rid, record = future_to_id[future]
                facts = future.result()
                
                if isinstance(facts, list):
                    for fact in facts:
                        fact['source_id'] = rid
                    new_facts.extend(facts)
                    
                    # 写入CSV
                    with open(self.config.facts_csv_path, "a", encoding="utf-8-sig", newline="") as f:
                        csv.writer(f).writerow([
                            rid,
                            json.dumps(record['conversation_history'], indent=2, ensure_ascii=False),
                            json.dumps(facts, indent=2, ensure_ascii=False)
                        ])
        
        return new_facts
    
    def save_evolutions(self, evolutions: List[Dict]):
        """保存记忆演变结果"""
        # 处理discarded IDs
        all_discarded = {i for evo in evolutions for i in evo.get("discarded_source_ids", [])}
        
        processed = [
            {**evo, "sorted_source_ids": [i for i in evo.get("sorted_source_ids", []) if i not in all_discarded]}
            for evo in evolutions
        ]
        
        with open(self.config.evolutions_csv_path, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["attribute_group", "sorted_source_ids", "discarded_source_ids", 
                           "narrative_summary", "reasoning", "original_facts"])
            for evo in processed:
                writer.writerow([
                    evo.get("attribute_group"),
                    evo.get("sorted_source_ids"),
                    json.dumps(evo.get("discarded_source_ids"), ensure_ascii=False),
                    evo.get("narrative_summary"),
                    evo.get("reasoning"),
                    json.dumps(evo.get("original_facts"), indent=2, ensure_ascii=False)
                ])
        
        print(f"记忆演变结果已保存到: {self.config.evolutions_csv_path}")
        print(f"共 {len(evolutions)} 个演变组，{len(all_discarded)} 个被丢弃的source ID")
    
    def run(self):
        """执行完整流程"""
        # Step 1: 加载记录
        print("\n=== Step 1: 加载记录 ===")
        records = self.load_records()
        
        # Step 2: 加载已有事实并提取新事实
        print("\n=== Step 2: 事实提取 ===")
        all_facts, processed_ids = self.load_existing_facts()
        new_facts = self.extract_facts_parallel(records, processed_ids)
        all_facts.extend(new_facts)
        
        # Step 3: 解析记忆演变
        print("\n=== Step 3: 记忆演变解析 ===")
        evolutions = self.resolver.resolve_all_groups(all_facts)
        self.save_evolutions(evolutions)
        
        print("\n处理完成!")


# ----------------------------
# 命令行参数解析
# ----------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="事实提取与记忆冲突检测工具")
    
    # 输入输出
    parser.add_argument(
        "--input-files", "-i", nargs="+",
        default=["toolace_formatted_conversations.jsonl", 
                 "oasst1_formatted_conversations.jsonl",
                 "bfcl_formatted_conversations.jsonl"],
        help="输入的JSONL文件列表"
    )
    parser.add_argument(
        "--output-dir", "-o", default="processed_data",
        help="输出目录 (default: processed_data)"
    )
    
    # 模型配置
    parser.add_argument(
        "--model", "-m", default="Qwen/Qwen3-Next-80B-A3B-Instruct",
        help="LLM模型名称"
    )
    parser.add_argument(
        "--embedding-model", default="BAAI/bge-m3",
        help="嵌入模型名称 (default: BAAI/bge-m3)"
    )
    
    # API配置
    parser.add_argument(
        "--api-base-url", default="https://api.siliconflow.cn/v1",
        help="API基础URL"
    )
    parser.add_argument(
        "--api-key", default=None,
        help="API密钥 (也可通过OPENAI_API_KEY环境变量设置)"
    )
    
    # 处理参数
    parser.add_argument(
        "--min-cluster-size", type=int, default=2,
        help="BERTopic最小聚类大小 (default: 2)"
    )
    parser.add_argument(
        "--max-workers", "-w", type=int, default=6,
        help="并行工作线程数 (default: 6)"
    )
    parser.add_argument(
        "--batch-size", "-b", type=int, default=64,
        help="嵌入计算批量大小 (default: 64)"
    )
    
    return parser.parse_args()


# ----------------------------
# 主入口
# ----------------------------
if __name__ == "__main__":
    args = parse_args()
    config = Config(args)
    pipeline = DataPipeline(config)
    pipeline.run()
