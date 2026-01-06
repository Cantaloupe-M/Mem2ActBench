import json

input_file = '/home/amax/syt/toolmem/toolace_formatted_conversations.jsonl'
output_file = '/home/amax/syt/toolmem/tool_api_pool.json'

tool_pool = set()

with open(input_file, 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line)
        if 'use_tools' in data:
            for tool in data['use_tools']:
                tool_pool.add(json.dumps(tool, sort_keys=True))

# Convert the set of JSON strings back to a list of dictionaries
unique_tools = [json.loads(tool) for tool in tool_pool]

with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(unique_tools, f, indent=4, ensure_ascii=False)

print(f'Extracted {len(unique_tools)} unique tools to {output_file}')