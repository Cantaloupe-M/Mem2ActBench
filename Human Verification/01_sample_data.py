import json
import random
import os
from collections import defaultdict

# Paths
INPUT_FILE = r'c:\Users\33109\OneDrive\Desktop\datagen\toolmem_dataset\processed_data\qa_dataset_graded.jsonl'
OUTPUT_FILE = r'c:\Users\33109\OneDrive\Desktop\datagen\toolmem_dataset\Human Verification\sampled_data.json'

def sample_data(input_path, output_path, sample_size_per_category=5):
    """
    Reads the dataset and samples items based on reasoning depth (Explicit/Semantic/Inferred)
    and dynamics (Static/Evolving).
    """
    if not os.path.exists(input_path):
        print(f"Error: Input file not found at {input_path}")
        return

    data_by_category = defaultdict(list)
    
    print(f"Reading from {input_path}...")
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                item = json.loads(line)
                
                # Extract stratification keys
                # Default to 'Unknown' if keys are missing
                reasoning = item.get('difficulty', {}).get('d2_reasoning', {}).get('label', 'Unknown')
                dynamics = item.get('difficulty', {}).get('d1_dynamics', {}).get('label', 'Unknown')
                
                # Simplify categories
                if 'Explicit' in reasoning:
                    cat_reasoning = 'Explicit'
                elif 'Semantic' in reasoning or 'Inferred' in reasoning:
                    cat_reasoning = 'Inferred'
                else:
                    cat_reasoning = 'Other'
                
                if 'Add' in dynamics:
                    cat_dynamics = 'Static'
                elif 'Refine' in dynamics or 'Conflict' in dynamics:
                    cat_dynamics = 'Evolving'
                else:
                    cat_dynamics = 'Other'
                
                category = f"{cat_dynamics}_{cat_reasoning}"
                data_by_category[category].append(item)
                
            except json.JSONDecodeError:
                continue

    sampled_items = []
    
    print("\nSampling Statistics:")
    for category, items in data_by_category.items():
        # If we have fewer items than requested, take all of them
        n = min(len(items), sample_size_per_category)
        sample = random.sample(items, n)
        sampled_items.extend(sample)
        print(f"  Category '{category}': Found {len(items)}, Sampled {n}")

    # Shuffle the final list to avoid bias in presentation order
    random.shuffle(sampled_items)
    
    print(f"\nTotal sampled items: {len(sampled_items)}")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(sampled_items, f, indent=2, ensure_ascii=False)
    print(f"Saved sampled data to {output_path}")

if __name__ == "__main__":
    # Adjust sample size as needed. For a real paper, this might be 50-100 per category.
    # For demonstration/code generation, we use 5.
    sample_data(INPUT_FILE, OUTPUT_FILE, sample_size_per_category=5)
