#!/usr/bin/env python
"""
Download and organize raw datasets to data/raw/

This script downloads the original raw files from:
- EmpatheticDialogues (HuggingFace)
- GoEmotions (HuggingFace)
- EmoryNLP (already downloaded from GitHub)
"""

import os
import sys
import json
from pathlib import Path
from tqdm import tqdm

# Set Chinese mirror
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets import load_dataset

def download_empathetic_raw(output_dir: str):
    """Download EmpatheticDialogues raw data"""
    print("\n" + "="*60)
    print("Downloading EmpatheticDialogues raw data")
    print("="*60)
    
    dataset_dir = Path(output_dir) / "empathetic_dialogues"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Download all splits
        for split in ["train", "validation", "test"]:
            print(f"\nDownloading {split} split...")
            dataset = load_dataset("empathetic_dialogues", split=split)
            
            # Save as JSON
            output_file = dataset_dir / f"{split}.json"
            data = []
            for item in tqdm(dataset, desc=f"Processing {split}"):
                data.append(item)
            
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            print(f"✓ Saved {len(data)} samples to {output_file}")
        
        print("\n✅ EmpatheticDialogues download complete!")
        return True
        
    except Exception as e:
        print(f"❌ Error downloading EmpatheticDialogues: {e}")
        return False


def download_goemotions_raw(output_dir: str):
    """Download GoEmotions raw data"""
    print("\n" + "="*60)
    print("Downloading GoEmotions raw data")
    print("="*60)
    
    dataset_dir = Path(output_dir) / "goemotions"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Download all splits
        for split in ["train", "validation", "test"]:
            print(f"\nDownloading {split} split...")
            dataset = load_dataset("go_emotions", split=split)
            
            # Save as JSON
            output_file = dataset_dir / f"{split}.json"
            data = []
            for item in tqdm(dataset, desc=f"Processing {split}"):
                data.append(item)
            
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            print(f"✓ Saved {len(data)} samples to {output_file}")
        
        print("\n✅ GoEmotions download complete!")
        return True
        
    except Exception as e:
        print(f"❌ Error downloading GoEmotions: {e}")
        return False


def check_emorynlp_raw(output_dir: str):
    """Check if EmoryNLP raw data exists"""
    print("\n" + "="*60)
    print("Checking EmoryNLP raw data")
    print("="*60)
    
    dataset_dir = Path(output_dir) / "emorynlp"
    
    if not dataset_dir.exists():
        print(f"❌ EmoryNLP directory not found: {dataset_dir}")
        print("Please download manually from:")
        print("  wget https://github.com/emorynlp/emotion-detection/tree/master/json")
        return False
    
    files = list(dataset_dir.glob("*.json"))
    if files:
        print(f"✅ EmoryNLP found: {len(files)} files")
        for f in files:
            print(f"  - {f.name}")
        return True
    else:
        print(f"❌ No JSON files found in {dataset_dir}")
        return False


def create_dataset_info(output_dir: str):
    """Create dataset info file"""
    info = {
        "datasets": {
            "empathetic_dialogues": {
                "source": "HuggingFace (empathetic_dialogues)",
                "description": "EmpatheticDialogues dataset with 32 emotion categories",
                "files": ["train.json", "validation.json", "test.json"],
                "url": "https://huggingface.co/datasets/empathetic_dialogues"
            },
            "goemotions": {
                "source": "HuggingFace (go_emotions)",
                "description": "GoEmotions dataset with 27 emotion categories",
                "files": ["train.json", "validation.json", "test.json"],
                "url": "https://huggingface.co/datasets/go_emotions"
            },
            "emorynlp": {
                "source": "GitHub (emorynlp/emotion-detection)",
                "description": "EmoryNLP emotion detection dataset",
                "files": ["emotion-detection-trn.json", "emotion-detection-dev.json", "emotion-detection-tst.json"],
                "url": "https://github.com/emorynlp/emotion-detection"
            }
        },
        "note": "All datasets are mapped to a unified 28-class emotion taxonomy"
    }
    
    info_file = Path(output_dir) / "dataset_info.json"
    with open(info_file, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Created dataset_info.json")


def main():
    # Use relative path from project root
    base_dir = Path(__file__).parent.parent
    output_dir = base_dir / "data" / "raw"
    
    print("="*60)
    print("Organizing Raw Emotion Recognition Datasets")
    print("="*60)
    print(f"Output directory: {output_dir}")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Download/Check each dataset
    results = {}
    
    # 1. EmpatheticDialogues
    if not (Path(output_dir) / "empathetic_dialogues").exists():
        results["empathetic_dialogues"] = download_empathetic_raw(output_dir)
    else:
        print("\n✅ EmpatheticDialogues already exists")
        results["empathetic_dialogues"] = True
    
    # 2. GoEmotions
    if not (Path(output_dir) / "goemotions").exists():
        results["goemotions"] = download_goemotions_raw(output_dir)
    else:
        print("\n✅ GoEmotions already exists")
        results["goemotions"] = True
    
    # 3. EmoryNLP
    results["emorynlp"] = check_emorynlp_raw(output_dir)
    
    # Create info file
    create_dataset_info(output_dir)
    
    # Summary
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    for dataset, success in results.items():
        status = "✅" if success else "❌"
        print(f"{status} {dataset}")
    
    print("\nDataset directory structure:")
    print(f"{output_dir}/")
    print("├── empathetic_dialogues/")
    print("│   ├── train.json")
    print("│   ├── validation.json")
    print("│   └── test.json")
    print("├── goemotions/")
    print("│   ├── train.json")
    print("│   ├── validation.json")
    print("│   └── test.json")
    print("├── emorynlp/")
    print("│   ├── emotion-detection-trn.json")
    print("│   ├── emotion-detection-dev.json")
    print("│   └── emotion-detection-tst.json")
    print("└── dataset_info.json")


if __name__ == "__main__":
    main()
