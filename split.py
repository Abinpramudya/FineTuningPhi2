import json
import random
import os

def split_jsonl_train_val(input_path, output_dir, train_ratio=0.8, seed=42):
    # Load all samples
    with open(input_path, 'r') as f:
        lines = f.readlines()

    # Shuffle the data
    random.seed(seed)
    random.shuffle(lines)

    # Split
    total = len(lines)
    train_end = int(train_ratio * total)

    train_data = lines[:train_end]
    val_data = lines[train_end:]

    # Make sure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Write files
    with open(os.path.join(output_dir, 'train.jsonl'), 'w') as f:
        f.writelines(train_data)

    with open(os.path.join(output_dir, 'valid.jsonl'), 'w') as f:
        f.writelines(val_data)

    print(f"Split complete:")
    print(f"Train: {len(train_data)} samples")
    print(f"Validation: {len(val_data)} samples")


split_jsonl_train_val("essential_dataset.jsonl", "split_output")

