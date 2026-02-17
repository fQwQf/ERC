#!/usr/bin/env python
"""
Data Preparation Script with Chinese Mirror Support

Downloads and preprocesses all emotion datasets using HF-Mirror.

Usage:
    python scripts/prepare_data_cn.py --config configs/config.yaml
"""

import os
import sys
import argparse
from pathlib import Path

# Set Chinese mirror before importing huggingface libraries
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.data_processor import DataProcessor
from omegaconf import OmegaConf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--datasets", type=str, nargs="+", 
                        default=["empathetic", "goemotions", "meld"])
    args = parser.parse_args()
    
    print("="*60)
    print("Data Preparation with HF-Mirror")
    print(f"HF_ENDPOINT: {os.environ.get('HF_ENDPOINT', 'Not set')}")
    print("="*60)
    
    # Load config
    config = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    
    # Initialize processor
    processor = DataProcessor(cache_dir=config["data"]["cache_dir"])
    
    # Process each split
    for split in ["train", "validation", "test"]:
        print(f"\n{'='*50}")
        print(f"Processing {split} split")
        print(f"{'='*50}")
        
        samples = processor.load_all_datasets(
            datasets=args.datasets,
            split=split,
        )
        
        if samples:
            # Print statistics
            processor.print_statistics(samples)
            
            # Save samples
            output_file = os.path.join(
                config["data"]["cache_dir"],
                f"{split}_samples.json"
            )
            processor.save_samples(samples, output_file)
        else:
            print(f"No samples found for {split} split")
    
    print("\n" + "="*50)
    print("Data preparation complete!")
    print("="*50)


if __name__ == "__main__":
    main()
