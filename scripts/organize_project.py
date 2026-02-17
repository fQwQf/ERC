#!/usr/bin/env python
"""
Organize project structure and create dataset summary
"""

import os
import json
from pathlib import Path
from collections import defaultdict

def organize_project():
    """Organize project structure"""
    
    # Get the directory where the script is located, then go up to root
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / "data"
    
    # Create directories
    (data_dir / "raw").mkdir(parents=True, exist_ok=True)
    (data_dir / "processed").mkdir(parents=True, exist_ok=True)
    (data_dir / "json").mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("PROJECT STRUCTURE")
    print("="*60)
    
    # Check existing datasets
    datasets = {}
    
    # Check EmpatheticDialogues & GoEmotions in cache
    cache_dir = base_dir / "cache" / "data"
    if cache_dir.exists():
        for file in cache_dir.glob("*.json"):
            print(f"✓ Found: {file.name}")
    
    # Check raw datasets
    raw_dir = data_dir / "raw"
    if raw_dir.exists():
        for subdir in raw_dir.iterdir():
            if subdir.is_dir():
                files = list(subdir.glob("*"))
                print(f"✓ Dataset: {subdir.name} ({len(files)} files)")
                datasets[subdir.name] = len(files)
    
    print("\n" + "="*60)
    print("DATASET SUMMARY")
    print("="*60)
    
    # Load and analyze processed data
    json_dir = data_dir / "json"
    for file in sorted(json_dir.glob("*.json")):
        try:
            with open(file) as f:
                data = json.load(f)
                if isinstance(data, list):
                    print(f"{file.name}: {len(data)} samples")
                elif isinstance(data, dict):
                    total = sum(len(v) for v in data.values() if isinstance(v, list))
                    print(f"{file.name}: {total} samples")
        except:
            print(f"{file.name}: unable to parse")
    
    print("\n" + "="*60)
    print("PROJECT STRUCTURE CREATED")
    print("="*60)
    print(f"Data directory: {data_dir}")
    print(f"  - raw/: Original downloaded datasets")
    print(f"  - processed/: Preprocessed data")
    print(f"  - json/: JSON format data")
    print("="*60)

if __name__ == "__main__":
    organize_project()
