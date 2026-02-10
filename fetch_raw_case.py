#!/usr/bin/env python3
"""
fetch_raw_iltur_case.py

Fetches a single raw court case from the IL-TUR HuggingFace dataset
and saves it as-is to a JSON file. No structuring, no processing.

Prerequisites:
    pip install datasets huggingface_hub

    You need a HuggingFace token with access to the gated dataset.
    1. Go to https://huggingface.co/datasets/Exploration-Lab/IL-TUR
    2. Accept the terms
    3. Set your token: export HF_TOKEN=hf_xxxxx
       (or run: huggingface-cli login)

Usage:
    python fetch_raw_iltur_case.py              # fetches case at index 0
    python fetch_raw_iltur_case.py 42           # fetches case at index 42
    python fetch_raw_iltur_case.py --id "xyz"   # fetches case by its ID field
"""

import sys
import json
import os

def main():
    from datasets import load_dataset

    token = os.environ.get("HF_TOKEN")

    print("Loading IL-TUR dataset (cjpe)...")
    ds = load_dataset("Exploration-Lab/IL-TUR", "cjpe", token=token)
    all_cases = ds["single_train"]
    print(f"Total cases: {len(all_cases)}")

    # Parse args
    if len(sys.argv) > 1 and sys.argv[1] == "--id":
        target_id = sys.argv[2]
        print(f"Searching for case ID: {target_id}")
        case = None
        for i in range(len(all_cases)):
            c = all_cases[i]
            if c["id"] == target_id:
                case = c
                break
        if case is None:
            print(f"Case ID '{target_id}' not found.")
            return
    else:
        idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
        print(f"Fetching case at index {idx}...")
        case = all_cases[idx]

    # Convert to plain dict (handles any HF-specific types)
    raw = {k: v for k, v in case.items()}

    out = "raw_iltur_case.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(raw, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Saved to: {out}")
    print(f"   Case ID:    {raw.get('id', 'N/A')}")
    print(f"   Label:      {raw.get('label', 'N/A')}")
    print(f"   Text length: {len(raw.get('text', '')):,} chars")
    print(f"   All keys:   {list(raw.keys())}")


if __name__ == "__main__":
    main()