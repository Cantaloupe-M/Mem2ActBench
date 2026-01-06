import os
import json
from typing import List
import hashlib
import csv
import time
from openai import OpenAI
from sentence_transformers import SentenceTransformer, util
from collections import defaultdict
from tqdm import tqdm
from itertools import combinations
from concurrent.futures import ThreadPoolExecutor, as_completed
import functools
import pickle
import threading
import numpy as np
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP
from hdbscan import HDBSCAN
import shutil
import sys
import torch
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.preprocessing import normalize

os.environ["OPENAI_BASE_URL"] = "https://api.siliconflow.cn/v1"
os.environ["OPENAI_API_KEY"] = "sk-omutivisxktqoglinfimxbckcxacgaqayusvamjquihiujov"
# os.environ["OPENAI_API_KEY"] = "sk-fLtrd6nJYPnyfnzRCf7aD4DdE7C04aCbA85bCb4cC2C1329a"
# os.environ["OPENAI_BASE_URL"] = "https://api.apiyi.com/v1"
# os.environ["OPENAI_BASE_URL"] = "http://localhost:11434/v1/"
# os.environ["OPENAI_API_KEY"] = "ollama"
# os.environ["NO_PROXY"] = "localhost,127.0.0.1"

# Initialize clients
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"), base_url=os.environ.get("OPENAI_BASE_URL"))
sbert_model = SentenceTransformer('BAAI/bge-m3')

if torch.cuda.is_available():
    sbert_model = sbert_model.to('cuda')
else:
    print("未检测到GPU，使用CPU模式")

# ----------------------------
# 模块一：事实提取 (No Changes)流程中的注释，使其更清楚地说明如何切换使用本地模型。
# ----------------------------

def extract_facts_with_llm(text_content):
    """Extracts facts, with caching functionality. Can use either API or local model."""
    with open('prompts/extraction_facts.md', 'r', encoding='utf-8') as file:
        prompt_template = file.read()
    prompt = prompt_template.format(text_content=text_content)

    max_retries = 10
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="Qwen/Qwen3-Next-80B-A3B-Instruct",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0
                # response_format={"type": "json_object"}
            )
            response_content = response.choices[0].message.content.replace("```json","").replace("```","")
            data = json.loads(response_content)
            return data
        except Exception as e:
            if attempt < max_retries - 1:
                delay = 2 ** attempt
                print(f"Fact extraction attempt {attempt+1}/{max_retries} failed: {e}, retrying in {delay}s...")
                time.sleep(delay)
            else:
                print(f"Fact extraction failed after {max_retries} attempts: {e}")
                return []

# ----------------------------
# 模块二：语义分组 (使用BERTopic)
# ----------------------------
def _group_semantically_similar_attributes(facts, min_cluster_size=2):
    """
    使用BERTopic对属性进行聚类分组
    
    Args:
        facts: 包含属性的事实列表
        min_cluster_size: 最小聚类大小
    
    Returns:
        属性到规范属性的映射字典
    """
    attrs = list({f["attribute"].strip().lower() for f in facts})
    if len(attrs) < min_cluster_size:
        return {a: a for a in attrs}

    
    # 使用GPU进行批量嵌入计算
    if torch.cuda.is_available():
        with torch.no_grad():
            # 根据数据量动态调整批量大小以避免内存溢出
            batch_size = min(64, max(8, len(attrs) // 10))  # 动态批量大小
            print(f"使用批量大小 {batch_size} 进行GPU嵌入计算")
            # 批量处理以提高GPU利用率
            embeddings = sbert_model.encode(attrs, batch_size=batch_size, show_progress_bar=True, convert_to_tensor=True)
            embeddings = embeddings.cpu().numpy()  # 转换为numpy数组供后续处理
            torch.cuda.empty_cache()  # 清空GPU缓存
    else:
        embeddings = sbert_model.encode(attrs, batch_size=128, show_progress_bar=True)
        # 对向量进行归一化处理
        embeddings = normalize(embeddings, norm='l2')
    

    hdbscan_model = HDBSCAN(
        min_cluster_size=2,
        metric='euclidean',  # 使用欧几里得距离
        cluster_selection_epsilon=0.01,
        cluster_selection_method='leaf',
        prediction_data=True
    )

    # 3. 初始化 BERTopic
    topic_model = BERTopic(
        embedding_model=None,
        umap_model=None,
        hdbscan_model=hdbscan_model,
        verbose=False
    )
    topics, probs = topic_model.fit_transform(attrs, embeddings)
    
    # 构建属性到规范属性的映射
    groups = {}
    topic_to_canonical = {}
    
    # 为每个主题选择一个规范属性（选择该主题中最具代表性的属性）
    for i, (attr, topic) in enumerate(zip(attrs, topics)):
        if topic == -1: 
            groups[attr] = attr
            continue
        if topic not in topic_to_canonical:
            topic_to_canonical[topic] = attr
        groups[attr] = topic_to_canonical[topic]
    
    return groups

# ----------------------------------------------------
# 模块三：记忆演变与排序 (NEW LOGIC)
# ----------------------------------------------------
def resolve_fact_group_with_llm(facts_group):
    """
    Analyzes a group of semantically similar facts to determine chronological order and memory evolution.
    """
    # Create a simplified representation for the prompt, which is cleaner and saves tokens.

    # Group facts by source_id, filtering out None values
    statements_by_source_id = defaultdict(list)
    for f in facts_group:
        source_id, facts = f.get("source_id"), f.get("facts")
        if source_id and facts:
            statements_by_source_id[source_id].append(facts)

    # Create a unique list of facts with combined statements per source_id
    simplified_facts = [
        {"source_id": source_id, "facts": ". ".join(facts)}
        for source_id, facts in statements_by_source_id.items()
    ]
    
    with open('prompts/resolve_fact_group原始.md', 'r', encoding='utf-8') as f:
        prompt = f.read()

    max_retries = 5
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="Qwen/Qwen3-Next-80B-A3B-Instruct",
                messages=[{"role": "user", "content": prompt.format(data=json.dumps(simplified_facts, indent=2, ensure_ascii=False))}],
                temperature=0.2
            )
            response_content = response.choices[0].message.content.replace("```json","").replace("```","")
            data = json.loads(response_content)
            all_group_ids = set(statements_by_source_id.keys())
            if set(data["sorted_source_ids"]) != all_group_ids - set(data["discarded_source_ids"]):
                raise ValueError("sorted_source_ids 与 (all_group_ids - discarded_source_ids) 不匹配，LLM 返回结果不完整或有误")
            else:
                return data
        except Exception as e:
            if attempt < max_retries - 1:
                delay = 2 ** attempt
                print(f"Group resolution attempt {attempt+1}/{max_retries} failed: {e}, retrying in {delay}s...")
                time.sleep(delay)
            else:
                print(f"Failed to resolve fact group after {max_retries} attempts: {e}")
                return None

def resolve_fact_groups(all_facts, min_cluster_size=2):
    """
    使用BERTopic对事实进行分组，并发送每个组到LLM进行时间排序和叙述构建。
    
    Args:
        all_facts: 所有事实的列表
        min_cluster_size: BERTopic聚类的最小簇大小
    
    Returns:
        已解析的事实组列表
    """
    if len(all_facts) < min_cluster_size:
        return []

    print("使用BERTopic对语义相似属性进行分组...")
    attribute_to_canonical_map = _group_semantically_similar_attributes(all_facts, min_cluster_size=min_cluster_size)
    
    grouped_by_semantic_attribute = defaultdict(list)
    for fact in all_facts:
        original_attr = fact["attribute"].strip().lower()
        canonical_attr = attribute_to_canonical_map.get(original_attr, original_attr)
        grouped_by_semantic_attribute[canonical_attr].append(fact)

    resolved_groups = []
    groups_to_process = [(attr, facts) for attr, facts in grouped_by_semantic_attribute.items()
                          if len(set(f["source_id"] for f in facts)) > 1]

    if not groups_to_process:
        print("没有找到包含多个事实的组进行解析。")
        return []
    
    print(f"找到{len(groups_to_process)}个组进行解析。使用LLM处理中...")
    with ThreadPoolExecutor(max_workers=4) as executor:
        future_to_group = {executor.submit(resolve_fact_group_with_llm, group[1]): group for group in groups_to_process}
        
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
# 模块四：主流程 (Updated with Checkpoint Mechanism)
# ----------------------------
if __name__ == "__main__":
    input_jsonl_fnames = ["toolace_formatted_conversations.jsonl", "oasst1_formatted_conversations.jsonl","bfcl_formatted_conversations.jsonl"]
    # 创建processed_data目录
    os.makedirs("processed_data", exist_ok=True)

    records = []
    for fname in input_jsonl_fnames:
        with open(fname, encoding="utf-8") as f:
            lines = f.readlines()
            if fname == "toolace_formatted_conversations.jsonl":
                records.extend(json.loads(line) for line in lines)
            else:
                # 读取全部
                records.extend(json.loads(line) for line in lines) #[:1000]

    print(f"Loaded {len(records)} records from input files.")
    
    # Step 1: Prepare user texts
    print("\nStep 1: Preparing user texts...")
    user_texts = []
    record_indices = []
    for i, record in enumerate(records):
        user_contents = [conv["content"] for conv in record.get("conversation_history", []) if conv.get("role") == "user" and conv.get("content")]
        if user_contents:
            user_texts.append("\n".join(user_contents))
            record_indices.append(i)

    # Step 2: Parallel fact extraction
    original_facts_csv_path = "processed_data/original_records_with_facts.csv"
    all_facts_with_source = []
    original_records_with_facts = []

    # 创建一个集合来存储已经处理过的 record_id
    processed_record_ids = set()
    
    if os.path.exists(original_facts_csv_path):
        print(f"\nStep 2: {original_facts_csv_path} found. Loading facts from CSV...")
        with open(original_facts_csv_path, "r", encoding="utf-8-sig") as f:
            reader = csv.reader(f)
            header = next(reader)  # Skip header
            for row in reader:
                record_id = row[0]
                original_text = row[1]
                extracted_facts = json.loads(row[2])
                original_records_with_facts.append({"record_id": record_id,"original_text": original_text,"extracted_facts": extracted_facts})
                # 所有事实抓取
                all_facts_with_source.extend(extracted_facts)
                # 将已处理的 record_id 添加到集合中
                processed_record_ids.add(record_id)
        print(f"Loaded {len(all_facts_with_source)} facts from {original_facts_csv_path}.")
    
    # 确定需要处理的记录索引4
    records_to_process = [idx for idx in record_indices if records[idx]['id'] not in processed_record_ids]
    
    if records_to_process:
        print(f"\nStep 2: Starting parallel fact extraction for {len(records_to_process)} records...")
        start_time = time.time()

        # 保存更新后的数据到 CSV 文件
        os.makedirs(os.path.dirname(original_facts_csv_path), exist_ok=True)
        if not os.path.exists(original_facts_csv_path):
            with open(original_facts_csv_path, "w", encoding="utf-8-sig", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["record_id", "conversation_history", "extracted_facts"])
            
        with ThreadPoolExecutor(max_workers = 6) as executor:
            future_to_idx = {executor.submit(extract_facts_with_llm, user_texts[record_indices.index(idx)]): idx for idx in records_to_process}
            
            for future in tqdm(as_completed(future_to_idx), total=len(records_to_process), desc="Extracting Facts"):
                idx = future_to_idx[future]
                facts = future.result()
                record_id = records[idx]['id']
                if isinstance(facts, list):
                    # 保存原始记录与facts的对应关系
                    original_records_with_facts.append({
                        "record_id": record_id,
                        "conversation_history": records[idx]['conversation_history'],
                        "extracted_facts": facts
                    })
                    for fact in facts:  
                        fact['source_id'] = record_id
                    records[idx]["attitude_facts"] = facts
                    all_facts_with_source.extend(facts) 
                    
                    # 边运行边写入CSV文件
                    with open(original_facts_csv_path, "a", encoding="utf-8-sig", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow([
                            record_id,
                            json.dumps(records[idx]['conversation_history'], indent=2, ensure_ascii=False),
                            json.dumps(facts, indent=2, ensure_ascii=False)
                        ])
                else:
                    print("user_texts:", user_texts[record_indices.index(idx)], "facts:", facts)

        extraction_time = time.time() - start_time
        print(f"Fact extraction complete. Extracted {len(records_to_process)} records in {extraction_time:.2f} seconds.")
    else:
        print("\nStep 2: All records have already been processed.")

    # Step 3: Resolve memory evolutions from the global fact pool iteratively
    print("\nStep 3: Starting iterative global memory evolution resolution...")
    discarded_ids_by_iteration = []
    min_cluster_size = 2 
    # 使用BERTopic处理当前活跃的事实集
    resolved_evolutions = resolve_fact_groups(all_facts_with_source, min_cluster_size=min_cluster_size) #[:200]
    
    all_discarded_ids = {i for evo in resolved_evolutions for i in evo.get("discarded_source_ids", [])}
    discarded_count = len(all_discarded_ids)
    print(f"Found {len(resolved_evolutions)} memory evolutions with {discarded_count} discarded source IDs")
    
    # Process resolved_evolutions to remove discarded IDs from kept_ids
    processed_evolutions = [
        {**evo, "sorted_source_ids": [i for i in evo.get("sorted_source_ids", []) if i not in all_discarded_ids]}
        for evo in resolved_evolutions
    ]
    
    iteration_prefix =  f"conflict_evolutions"
    # Save memory_evolutions.csv for this iteration
    with open(f"processed_data/{iteration_prefix}.csv", "w", encoding="utf-8-sig", newline="") as f:   
        writer = csv.writer(f)
        writer.writerow(["attribute_group","sorted_source_ids","discarded_source_ids","narrative_summary","reasoning","original_facts"])
        for evo in processed_evolutions:
            writer.writerow([
                evo.get("attribute_group"),
                evo.get("sorted_source_ids"),
                json.dumps(evo.get("discarded_source_ids"), ensure_ascii=False),
                evo.get("narrative_summary"),
                evo.get("reasoning"),
                json.dumps(evo.get("original_facts"), indent=2, ensure_ascii=False)
            ])
    print(f"Iteration memory evolutions saved to: {iteration_prefix}.csv")
