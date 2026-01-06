from collections import defaultdict, deque
from typing import Any
import pandas as pd
import json
import random
import ast
from openai import OpenAI
import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import pickle

os.environ["OPENAI_BASE_URL"] = "https://api.siliconflow.cn/v1"
os.environ["OPENAI_API_KEY"] = "sk-omutivisxktqoglinfimxbckcxacgaqayusvamjquihiujov"
# os.environ["OPENAI_API_KEY"] = "sk-fLtrd6nJYPnyfnzRCf7aD4DdE7C04aCbA85bCb4cC2C1329a"
# os.environ["OPENAI_BASE_URL"] = "https://api.apiyi.com/v1"
# os.environ["OPENAI_BASE_URL"] = "http://localhost:11434/v1/"
# os.environ["OPENAI_API_KEY"] = "ollama"
# os.environ["NO_PROXY"] = "localhost,127.0.0.1"

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"), base_url=os.environ.get("OPENAI_BASE_URL"))

def find_interleaved_sequence(sequences):
    """
    使用【优化后的】拓扑排序来合并序列。
    当检测到冲突（循环）时，它会使用启发式策略（丢弃出度最高的节点）
    来尝试最小化被丢弃的节点总数。
    
    参数:
    sequences (list of list): 包含多个序列的列表。
    
    返回:
    (result, discarded_nodes): 
        - result (list): 一个合并后的、尽可能满足约束的序列。
        - discarded_nodes (list): 因冲突而被丢弃的节点列表。
    """
    
    adj_list = defaultdict(set)
    in_degree = defaultdict(int)
    all_nodes = set()

    # --- 1. 构建图和计算入度 ---
    for seq in sequences:
        if not seq:
            continue
        all_nodes.update(seq)
        
        for i in range(len(seq) - 1):
            u = seq[i]
            v = seq[i+1]
            if v not in adj_list[u]:
                adj_list[u].add(v)
                in_degree[v] += 1

    # --- 2. 初始化队列 ---
    # 移除 sorted()
    zero_in_degree_nodes = [node for node in all_nodes if in_degree[node] == 0]
    # 增加随机性
    random.shuffle(zero_in_degree_nodes)
    queue = deque(zero_in_degree_nodes)
        
    result = [] 
    discarded_nodes = [] 
    processed_count = 0

    # --- 3. 执行修改后的Kahn算法 ---
    while processed_count < len(all_nodes):
        
        if queue:
            # --- a. 正常处理 (队列非空) ---
            u = queue.popleft()
            result.append(u)
            processed_count += 1
            
            for v in sorted(list(adj_list[u])): # 按字典序迭代，确保可复现性
                in_degree[v] -= 1
                if in_degree[v] == 0:
                    queue.append(v)
        else:
            # --- b. 冲突处理 (队列为空，但仍有节点未处理) ---
            # 这意味着我们遇到了一个或多个环
            # 找到所有“卡住”的节点
            stuck_nodes = [
                node for node in all_nodes 
                if node not in result and node not in discarded_nodes
            ]
            if not stuck_nodes:
                break # 没有卡住的节点了，安全退出

            # ========================================================
            # === 启发式优化：选择“问题最大”的节点 ===
            # ========================================================
            # 策略：选择“卡住”的节点中 "out-degree" (出度) 最高的节点。
            # 丢弃它会“解锁”最多的后续节点。
            # 如果出度相同，我们选择字典序最小的那个，以保证结果的确定性。
            
            node_to_discard = None
            max_out_degree = -1

            for node in stuck_nodes:
                # 获取该节点的出度（它指向多少个其他节点）
                current_out_degree = len(adj_list[node])
                
                if current_out_degree > max_out_degree:
                    # 找到了一个出度更高的节点
                    max_out_degree = current_out_degree
                    node_to_discard = node
                elif current_out_degree == max_out_degree:
                    # 出度相同，用字典序来打破平局，确保一致性
                    if node_to_discard is None or node < node_to_discard:
                        node_to_discard = node
            # ========================================================

            if node_to_discard is None:
                # 理论上不应该发生，但作为安全退出
                break 

            print(f"检测到冲突（循环），启发式丢弃节点 {node_to_discard} (出度: {max_out_degree}) 以继续...")
            discarded_nodes.append(node_to_discard)
            processed_count += 1
            
            # "丢弃"这个节点：释放c它的邻居
            for v in sorted(list(adj_list[node_to_discard])):
                in_degree[v] -= 1
                if in_degree[v] == 0:
                    queue.append(v)

    # --- 4. 检查冲突（循环依赖） ---
    # 如果最终结果的长度小于总节点数，说明图中存在环
    if len(result) < len(all_nodes):
        # 找出那些因为循环而无法被排序的节点
        conflicted_nodes = all_nodes - set(result)
        print(f"检测到冲突（循环依赖），无法完成排序。")
        print(f"涉及冲突的节点（或受其影响的节点）: {sorted(list(conflicted_nodes))}")
        return result, conflicted_nodes
    else:
        return result, set()

def extract_key_points(sequences_input_filtered, all_facts_df, checkpoint_file='processed_data/checkpoint.pkl'):
    """
    调用大模型抓取主要记忆点
    支持断点恢复机制
    """
    def process_sequence(seq):
        # 将seq转换为事实组列表
        facts = []
        for source_id in seq:
            # 从all_facts_df中查找对应的fact
            fact_row = all_facts_df[all_facts_df['record_id'] == source_id]
            if not fact_row.empty:
                fact = {
                    "source_id": source_id,
                    "fact": fact_row['extracted_facts'].iloc[0]
                }
                facts.append(fact)
        
        if not facts:
            return []
    
        with open("prompts/memory_evolution.md", "r", encoding="utf-8") as f:
            prompt_template = f.read()
        data_str = json.dumps(facts, indent=2, ensure_ascii=False)
        prompt = prompt_template.format(data=data_str)
        
        # 调用大模型，带指数退避重试机制
        max_retries = 10
        base_delay = 1  # 初始延迟时间（秒）
        
        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model="moonshotai/Kimi-K2-Instruct-0905",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                )
                
                # 解析响应
                response_content = response.choices[0].message.content.replace('```json','').replace('```','')
                try:
                    data_list = json.loads(response_content)
                    result_list = []
                    for data in data_list:
                        seen = set() #去重
                        sub_ids = []
                        for item in data["evolution_chain"]:
                            sid = item["source_id"]
                            if sid not in seen:
                                seen.add(sid)
                                sub_ids.append(sid)
                        data["sub_ids"] = ",".join(sub_ids)
                        data["group_ids"] = ",".join(seq)
                        result_list.append(data)
                    return result_list
                except json.JSONDecodeError:
                    print(f"JSON解析错误: {response_content}")
                    return []
            except Exception as e:
                print(f"Error calling LLM: {e}")
                if attempt < max_retries - 1:  # 如果不是最后一次尝试，则等待后重试
                    delay = base_delay * (2 ** attempt)  # 指数退避
                    print(f"等待 {delay} 秒后进行第 {attempt + 2} 次重试...")
                    time.sleep(delay)
                else:
                    return []
    
    # === 断点恢复机制 ===
    # 1. 尝试加载之前的检查点
    checkpoint_data = {}
    processed_indices = set()
    event_groups_list = []
    
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, 'rb') as f:
                checkpoint_data = pickle.load(f)
                processed_indices = checkpoint_data.get('processed_indices', set())
                event_groups_list = checkpoint_data.get('results', [])
                print(f"从断点恢复: 已处理 {len(processed_indices)}/{len(sequences_input_filtered)} 个序列")
        except Exception as e:
            print(f"加载检查点失败: {e}，从头开始处理")
            processed_indices = set()
            event_groups_list = []
    
    # 2. 过滤出未处理的序列
    sequences_to_process = [(i, seq) for i, seq in enumerate(sequences_input_filtered) if i not in processed_indices]
    
    if not sequences_to_process:
        print("所有序列已处理完成！")
        return event_groups_list
    
    print(f"剩余 {len(sequences_to_process)} 个序列待处理")
    
    # 3. 使用线程池处理剩余序列
    with ThreadPoolExecutor(max_workers=4) as executor:  # 可以根据需要调整线程数
        future_to_index = {executor.submit(process_sequence, seq): (idx, seq) for idx, seq in sequences_to_process}
        
        for future in tqdm(as_completed(future_to_index), total=len(sequences_to_process), desc="Processing sequences"):
            try:
                idx, seq = future_to_index[future]
                event_groups = future.result()
                event_groups_list.append(event_groups)
                processed_indices.add(idx)
                
                # 4. 定期保存检查点（每处理完一个序列就保存）
                checkpoint_data = {
                    'processed_indices': processed_indices,
                    'results': event_groups_list
                }
                with open(checkpoint_file, 'wb') as f:
                    pickle.dump(checkpoint_data, f)
                    
            except Exception as exc:
                print(f'Generated an exception: {exc}')
    
    # 5. 处理完成后删除检查点文件
    if os.path.exists(checkpoint_file):
        try:
            os.remove(checkpoint_file)
            print("所有序列处理完成，检查点文件已删除")
        except Exception as e:
            print(f"删除检查点文件失败: {e}")
    
    return event_groups_list
    
def deduplicate_sequence(seq):
    """
    去除序列中重复的id，仅保留最后一次出现的元素。
    """
    seen = set()
    result = []
    for item in reversed(seq):
        if item not in seen:
            seen.add(item)
            result.append(item)
    result.reverse()
    return result


if __name__ == "__main__":
    # 读取序列
    final_groups = pd.read_csv("processed_data/conflict_evolutions.csv")
    sequences_input = [list[Any](ast.literal_eval(kept_ids_str)) for kept_ids_str in final_groups['sorted_source_ids']]
    all_facts_df = pd.read_csv('processed_data/original_records_with_facts.csv')
    random.shuffle(sequences_input) 
    print(f"开始处理 {len(sequences_input)} 条总序列...")
    result_final, dropped = find_interleaved_sequence(sequences_input)

    print(f"排序完成。最终序列长度: {len(result_final)}, 丢弃节点数: {len(dropped)}")
    sequences_input_filtered = [[x for x in seq if x not in dropped] for seq in sequences_input]
    # 根据排序后的 id 列表，从 all_facts_df 中取出对应的 conversation_history
    merged_df = (
        pd.DataFrame({'id': result_final})
          .merge(all_facts_df[['record_id', 'conversation_history']], 
                 left_on='id', right_on='record_id', how='left')
          .drop(columns=['record_id'])          # 去掉多余的 record_id 列
    )
    merged_df.to_csv('processed_data/conversation_sequence.csv', index=False)
    
    final_event_groups = extract_key_points(sequences_input_filtered, all_facts_df)
    
    # 保存为Json
    with open('processed_data/memory_key_points.json', 'w', encoding='utf-8') as f:
        json.dump(final_event_groups, f, ensure_ascii=False, indent=2)
    
    # 保存为csv
    # 将 final_event_groups（列表的列表）展平为字典列表
    # flat_list = [item for sublist in final_event_groups for item in sublist]
    # for item in flat_list:
    #     item['evolution_chain'] = json.dumps(item['evolution_chain'], indent=2,ensure_ascii=False)

    # df_keypoints = pd.DataFrame(flat_list)
    # df_keypoints.to_csv('processed_data/memory_key_points.csv', index=False, encoding='utf-8')
    
    print(f"最终组数: {len(final_event_groups)}")
    print(f"总体序列长度: {len(result_final)}")