import json, pandas as pd
from tqdm import tqdm
from deep_translator import GoogleTranslator
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict


def translate_to_english(text: str) -> str:
    """
    使用 deep_translator 库中的 GoogleTranslator 将文本翻译成英文。
    """
    try:
        translated_text = GoogleTranslator(source='auto', target='en').translate(text)
        return translated_text
    except Exception as e:
        print(f"翻译失败: {e}. 返回原始文本。")
        return text

def translate_conv(conv: list) -> list:
    """单条对话并行翻译所有 turn"""
    for turn in conv:
        turn["content"] = translate_to_english(turn["content"])
    return conv

FILE = r"original_data/oasst1/2023-04-12_oasst_ready.messages.jsonl"

records = []
with open(FILE, "rt", encoding="utf-8") as f:
    for line in f:
        records.append(json.loads(line))
df = pd.DataFrame(records)

msg_by_id = df.set_index("message_id").to_dict("index")
best_assist = df.query("role == 'assistant' and rank == 0").copy()
# best_assist = best_assist.iloc[:50]

conversations = []         
for _, row in tqdm(best_assist.iterrows(), total=len(best_assist)):

    path = []
    cur = row
    # 4) 递归向上找 parent，直到 root（parent_id 为 None）
    while cur is not None:
        turn_data = {
            "role": "user" if cur["role"] == "prompter" else "assistant",
            "content": cur.get("text", "")
        }
        
        path.append(turn_data)
        pid = cur["parent_id"]
        cur = msg_by_id.get(pid)
    conversations.append(list(reversed(path)))


prompt2convs = defaultdict(list)
for conv in conversations:
    first_prompt = conv[0]["content"]
    prompt2convs[first_prompt].append(conv)

deduped = []
for prompt, convs in prompt2convs.items():
    # 按对话轮数降序，取第一条
    best = max(convs, key=lambda c: len(c))
    deduped.append(best)
conversations = deduped

# conversations = conversations[:50]
# # 并行翻译所有对话
with ThreadPoolExecutor(max_workers=32) as pool:
    futures = [pool.submit(translate_conv, c) for c in conversations]
    conversations = [f.result() for f in tqdm(futures, total=len(conversations), desc="Translating conversations")]

output_jsonl_fname = "oasst1_formatted_conversations.jsonl"
with open(output_jsonl_fname, "w", encoding="utf-8") as f:
    for idx, conv in enumerate(conversations):
        final_record = {
            "id": f"oss_{idx}",
            "conversation_history": conv,
        }
        f.write(json.dumps(final_record, ensure_ascii=False) + "\n")

print(f"Formatted conversations saved to {output_jsonl_fname}")
