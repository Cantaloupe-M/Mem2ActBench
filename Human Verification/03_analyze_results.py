import csv
import os
import sys
from collections import Counter
try:
    from sklearn.metrics import cohen_kappa_score
except ImportError:
    print("Warning: sklearn not found. Cohen's Kappa will not be calculated.")
    cohen_kappa_score = None

# Default Paths (Can be changed)
ANNOTATOR_1_FILE = r'c:\Users\33109\OneDrive\Desktop\datagen\toolmem_dataset\Human Verification\annotator1.csv'
ANNOTATOR_2_FILE = r'c:\Users\33109\OneDrive\Desktop\datagen\toolmem_dataset\Human Verification\annotator2.csv'

METRIC_COLUMNS = {
    'Tool Correctness': '[Check 1] Tool Correctness (0/1)',
    'Memory Dependency': '[Check 2] Memory Dependency (0/1)',
    'Argument Accuracy': '[Check 3] Argument Accuracy (1-5)',
    'Query Naturalness': '[Check 4] Query Naturalness (1-5)'
}

def read_csv(filepath):
    """Reads the annotated CSV and returns a dict of {sample_id: scores_dict}."""
    if not os.path.exists(filepath):
        return None
    
    data = {}
    with open(filepath, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            sample_id = row['Sample_ID']
            scores = {}
            for metric, col_name in METRIC_COLUMNS.items():
                val = row.get(col_name, '').strip()
                if val:
                    try:
                        scores[metric] = int(val)
                    except ValueError:
                        scores[metric] = None
            data[sample_id] = scores
    return data

def analyze_single_file(data, filename):
    """Analyzes the distribution of scores for a single annotator."""
    print(f"\n=== Analysis for {filename} ===")
    total = len(data)
    if total == 0:
        print("No data found.")
        return

    for metric in METRIC_COLUMNS.keys():
        scores = [d[metric] for d in data.values() if d.get(metric) is not None]
        if not scores:
            print(f"{metric}: No scores provided.")
            continue
        
        avg = sum(scores) / len(scores)
        counts = Counter(scores)
        
        print(f"\n{metric}:")
        print(f"  Average: {avg:.2f}")
        print(f"  Distribution: {dict(sorted(counts.items()))}")
        
        if metric == 'Memory Dependency':
            print(f"  Dependency Rate (Score=1): {counts.get(1, 0) / len(scores) * 100:.1f}%")
        if metric == 'Tool Correctness':
            print(f"  Correctness Rate (Score=1): {counts.get(1, 0) / len(scores) * 100:.1f}%")

def calculate_agreement(data1, data2):
    """Calculates Inter-Annotator Agreement (Cohen's Kappa)."""
    print(f"\n=== Inter-Annotator Agreement ===")
    
    common_ids = set(data1.keys()) & set(data2.keys())
    print(f"Common samples annotated: {len(common_ids)}")
    
    if len(common_ids) == 0:
        return

    for metric in METRIC_COLUMNS.keys():
        y1 = []
        y2 = []
        for sid in common_ids:
            v1 = data1[sid].get(metric)
            v2 = data2[sid].get(metric)
            if v1 is not None and v2 is not None:
                y1.append(v1)
                y2.append(v2)
        
        if not y1:
            continue
            
        if cohen_kappa_score:
            kappa = cohen_kappa_score(y1, y2)
            print(f"{metric}: Cohen's Kappa = {kappa:.4f}")
        
        # Calculate simple agreement rate
        agree_count = sum(1 for a, b in zip(y1, y2) if a == b)
        agree_rate = agree_count / len(y1)
        print(f"{metric}: Exact Match Rate = {agree_rate:.2%}")

if __name__ == "__main__":
    # Check if specific files were passed as args
    files = sys.argv[1:]
    if not files:
        # Use defaults or try to find generated csv
        generated_csv = r'c:\Users\33109\OneDrive\Desktop\datagen\toolmem_dataset\Human Verification\annotation_sheet.csv'
        if os.path.exists(generated_csv):
            print("No arguments provided. Analyzing the generated annotation sheet (assuming it might have some data)...")
            data = read_csv(generated_csv)
            analyze_single_file(data, "annotation_sheet.csv")
            print("\nTo calculate agreement, provide two CSV files as arguments.")
        else:
            print("No CSV files found to analyze. Run 02_generate_annotation_interface.py first.")
    elif len(files) == 1:
        data = read_csv(files[0])
        if data:
            analyze_single_file(data, files[0])
    elif len(files) >= 2:
        data1 = read_csv(files[0])
        data2 = read_csv(files[1])
        if data1 and data2:
            analyze_single_file(data1, files[0])
            analyze_single_file(data2, files[1])
            calculate_agreement(data1, data2)
