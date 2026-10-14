import argparse
from pathlib import Path
from collections import defaultdict
from typing import Any
import pandas as pd
import json
import random
import ast
import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import pickle


def find_interleaved_sequence(sequences):
    """
    使用优化后的拓扑排序来合并序列。
    当检测到冲突（循环）时，使用启发式策略（丢弃出度最高的节点）来最小化被丢弃的节点总数。
    """
    adj_list = defaultdict(set)
    in_degree = defaultdict(int)
    all_nodes = set()

    for seq in sequences:
        if not seq:
            continue
        all_nodes.update(seq)
        for i in range(len(seq) - 1):
            u, v = seq[i], seq[i+1]
            if v not in adj_list[u]:
                adj_list[u].add(v)
                in_degree[v] += 1

    zero_in_degree_nodes = [node for node in all_nodes if in_degree[node] == 0]
    random.shuffle(zero_in_degree_nodes)
    queue = zero_in_degree_nodes
        
    result = [] 
    discarded_nodes = [] 
    processed_count = 0

    while processed_count < len(all_nodes):
        if queue:
            u = queue.pop(0)
            result.append(u)
            processed_count += 1
            
            for v in sorted(list(adj_list[u])):
                in_degree[v] -= 1
                if in_degree[v] == 0:
                    queue.append(v)
        else:
            stuck_nodes = [
                node for node in all_nodes 
                if node not in result and node not in discarded_nodes
            ]
            if not stuck_nodes:
                break

            node_to_discard = None
            max_out_degree = -1

            for node in stuck_nodes:
                current_out_degree = len(adj_list[node])
                if current_out_degree > max_out_degree:
                    max_out_degree = current_out_degree
                    node_to_discard = node
                elif current_out_degree == max_out_degree:
                    if node_to_discard is None or node < node_to_discard:
                        node_to_discard = node

            if node_to_discard is None:
                break 

            print(f"检测到冲突（循环），启发式丢弃节点 {node_to_discard} (出度: {max_out_degree}) 以继续...")
            discarded_nodes.append(node_to_discard)
            processed_count += 1
            
            for v in sorted(list(adj_list[node_to_discard])):
                in_degree[v] -= 1
                if in_degree[v] == 0:
                    queue.append(v)

    if len(result) < len(all_nodes):
        conflicted_nodes = all_nodes - set(result)
        print(f"检测到冲突（循环依赖），无法完成排序。")
        print(f"涉及冲突的节点（或受其影响的节点）: {sorted(list(conflicted_nodes))}")
        return result, conflicted_nodes
    else:
        return result, set()

def main():
    parser = argparse.ArgumentParser(description="合并记忆演化链并排序对话")
    parser.add_argument("--evolutions", type=Path, default=Path("processed_data/conflict_evolutions.csv"),
                        help="记忆演化 CSV 路径")
    parser.add_argument("--facts", type=Path, default=Path("processed_data/original_records_with_facts.csv"),
                        help="事实记录 CSV 路径")
    parser.add_argument("-o", "--output", type=Path, default=Path("processed_data/conversation_sequence.csv"),
                        help="排序后的对话 CSV 路径（覆盖写入）")
    args = parser.parse_args()

    final_groups = pd.read_csv(args.evolutions)
    sequences_input = [list[Any](ast.literal_eval(kept_ids_str)) for kept_ids_str in final_groups['sorted_source_ids']]
    all_facts_df = pd.read_csv(args.facts)
    random.shuffle(sequences_input) 
    print(f"开始处理 {len(sequences_input)} 条总序列...")
    result_final, dropped = find_interleaved_sequence(sequences_input)

    print(f"排序完成。最终序列长度: {len(result_final)}, 丢弃节点数: {len(dropped)}")
    sequences_input_filtered = [[x for x in seq if x not in dropped] for seq in sequences_input]
    
    merged_df = (
        pd.DataFrame({'id': result_final})
          .merge(all_facts_df[['record_id', 'conversation_history']], 
                 left_on='id', right_on='record_id', how='left')
          .drop(columns=['record_id'])
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    merged_df.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
