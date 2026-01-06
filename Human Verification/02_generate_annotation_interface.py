import json
import csv
import os

INPUT_FILE = r'c:\Users\33109\OneDrive\Desktop\datagen\toolmem_dataset\Human Verification\sampled_data.json'
OUTPUT_FILE = r'c:\Users\33109\OneDrive\Desktop\datagen\toolmem_dataset\Human Verification\annotation_sheet.csv'

def format_memory(evolution_chain):
    """Formats the memory evolution chain into a readable string."""
    lines = []
    for i, entry in enumerate(evolution_chain, 1):
        fact = entry.get('fact', '')
        source_id = entry.get('source_id', 'unknown')
        evolution_type = entry.get('evolution_type', 'unknown')
        lines.append(f"[{i}] ({evolution_type}) {fact}")
    return "\n".join(lines)

def format_tool_calls(tool_calls):
    """Formats tool calls into a readable string."""
    if not tool_calls:
        return "No Tool Call"
    
    lines = []
    for call in tool_calls:
        name = call.get('name', 'UnknownTool')
        args = json.dumps(call.get('arguments', {}), ensure_ascii=False)
        lines.append(f"Tool: {name}\nArgs: {args}")
        
        grounding = call.get('grounding_info', {})
        if grounding:
            lines.append("Grounding:")
            for key, info in grounding.items():
                source_text = info.get('source_text', 'N/A')
                g_type = info.get('type', 'N/A')
                lines.append(f"  - {key} ({g_type}): {source_text}")
    
    return "\n".join(lines)

def generate_csv(input_path, output_path):
    if not os.path.exists(input_path):
        print(f"Error: Input file not found at {input_path}")
        return

    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Define CSV headers
    headers = [
        'Sample_ID',
        'Thread_Topic',
        'Memory_Context',
        'User_Query',
        'Model_Generated_Tool_Call',
        'Reasoning_Level',
        'Dynamics_Type',
        # Evaluation Columns (Empty for annotators)
        '[Check 1] Tool Correctness (0/1)',
        '[Check 2] Memory Dependency (0/1)',
        '[Check 3] Argument Accuracy (1-5)',
        '[Check 4] Query Naturalness (1-5)',
        'Comments'
    ]

    print(f"Generating CSV with {len(data)} items...")
    
    with open(output_path, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        
        for item in data:
            row = [
                item.get('thread_id', 'Unknown_ID'),
                item.get('thread_topic', ''),
                format_memory(item.get('evolution_chain', [])),
                item.get('qa_pair', {}).get('query', ''),
                format_tool_calls(item.get('qa_pair', {}).get('tool_calls', [])),
                item.get('difficulty', {}).get('d2_reasoning', {}).get('label', ''),
                item.get('difficulty', {}).get('d1_dynamics', {}).get('label', ''),
                '', '', '', '', '' # Empty columns for annotation
            ]
            writer.writerow(row)

    print(f"CSV generated at {output_path}")

if __name__ == "__main__":
    generate_csv(INPUT_FILE, OUTPUT_FILE)
