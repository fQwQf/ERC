#!/usr/bin/env python
"""
Training Script for Fine-Grained Emotion Recognition

Usage:
    python train.py --config configs/config.yaml
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, TaskType
import datasets
from omegaconf import OmegaConf
from tqdm import tqdm

from src.data.data_processor import DataProcessor, DialogueSample
from src.data.dataset import EmotionDataset, EmotionDatasetConfig, collate_fn
from src.data.emotion_taxonomy import TAXONOMY
from src.models.model import EmotionRecognitionModel, ModelConfig
from src.retrieval.retriever import Retriever, BatchRetriever


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    config = OmegaConf.load(config_path)
    return OmegaConf.to_container(config, resolve=True)


def prepare_data(config: dict) -> tuple:
    """Prepare training and validation data"""
    processor = DataProcessor(cache_dir=config["data"]["cache_dir"])
    
    # Load or process data
    train_file = config["data"]["train_file"]
    val_file = config["data"]["val_file"]
    
    if os.path.exists(train_file) and os.path.exists(val_file):
        print("Loading cached data...")
        train_samples = processor.load_samples(train_file)
        val_samples = processor.load_samples(val_file)
    else:
        print("Processing datasets...")
        # Load all datasets
        train_samples = processor.load_all_datasets(
            datasets=config["data"]["datasets"],
            split="train"
        )
        
        # For validation, use a subset or separate validation split
        val_samples = processor.load_all_datasets(
            datasets=config["data"]["datasets"],
            split="validation"
        )
        
        # Save for caching
        os.makedirs(os.path.dirname(train_file), exist_ok=True)
        processor.save_samples(train_samples, train_file)
        processor.save_samples(val_samples, val_file)
    
    # Print statistics
    processor.print_statistics(train_samples)
    
    return train_samples, val_samples


def build_retriever(samples: list, config: dict, device: str = "cuda") -> Retriever:
    """Build retrieval index for training data"""
    retriever = Retriever(
        model_name=config["retrieval"]["embedding_model"],
        cache_dir=config["retrieval"]["cache_dir"],
        device=device,
    )
    
    index_path = os.path.join(config["retrieval"]["cache_dir"], "index")
    
    if os.path.exists(index_path):
        print("Loading existing retrieval index...")
        retriever.load_index(index_path)
    else:
        print("Building retrieval index...")
        retriever.build_index(samples)
        retriever.save_index(index_path)
    
    return retriever


def create_datasets(
    train_samples: list,
    val_samples: list,
    tokenizer,
    config: dict,
    retriever: Retriever = None,
) -> tuple:
    """Create PyTorch datasets"""
    dataset_config = EmotionDatasetConfig(
        max_length=config["model"]["max_length"],
        use_retrieval=config["retrieval"]["top_k"] > 0,
        top_k_demonstrations=config["retrieval"]["top_k"],
    )
    
    # Retrieve examples for training set
    train_retrieved = None
    val_retrieved = None
    
    if retriever and dataset_config.use_retrieval:
        print("Retrieving demonstrations for training data...")
        batch_retriever = BatchRetriever(retriever, k=dataset_config.top_k_demonstrations)
        train_retrieved = batch_retriever.retrieve_batch(train_samples[:1000])  # Limit for speed
        
        print("Retrieving demonstrations for validation data...")
        val_retrieved = batch_retriever.retrieve_batch(val_samples[:200])
    
    train_dataset = EmotionDataset(
        samples=train_samples,
        tokenizer=tokenizer,
        config=dataset_config,
        retrieved_examples=train_retrieved,
    )
    
    val_dataset = EmotionDataset(
        samples=val_samples,
        tokenizer=tokenizer,
        config=dataset_config,
        retrieved_examples=val_retrieved,
    )
    
    return train_dataset, val_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--local_rank", type=int, default=-1)
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Set up output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(config["training"]["output_dir"], f"run_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save config
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"Output directory: {output_dir}")
    
    # Prepare data
    train_samples, val_samples = prepare_data(config)
    
    # Build retriever
    device = "cuda" if torch.cuda.is_available() else "cpu"
    retriever = build_retriever(train_samples, config, device)
    
    # Load model and tokenizer
    print(f"Loading model: {config['model']['base_model']}")
    
    model_wrapper = EmotionRecognitionModel(
        config=ModelConfig(
            base_model_name=config["model"]["base_model"],
            lora_r=config["lora"]["r"],
            lora_alpha=config["lora"]["lora_alpha"],
            lora_dropout=config["lora"]["lora_dropout"],
            target_modules=config["lora"]["target_modules"],
            use_flash_attention=config["model"]["use_flash_attention"],
        ),
        device_map="auto",
    )
    model_wrapper.load_base_model()
    model_wrapper.setup_lora()
    
    model = model_wrapper.get_model()
    tokenizer = model_wrapper.get_tokenizer()
    
    # Create datasets
    train_dataset, val_dataset = create_datasets(
        train_samples, val_samples, tokenizer, config, retriever
    )
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Val dataset size: {len(val_dataset)}")
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=config["training"]["num_train_epochs"],
        per_device_train_batch_size=config["training"]["per_device_train_batch_size"],
        per_device_eval_batch_size=config["training"]["per_device_eval_batch_size"],
        gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"],
        learning_rate=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
        warmup_ratio=config["training"]["warmup_ratio"],
        lr_scheduler_type=config["training"]["lr_scheduler_type"],
        logging_dir=os.path.join(output_dir, "logs"),
        logging_steps=config["training"]["logging_steps"],
        save_steps=config["training"]["save_steps"],
        eval_steps=config["training"]["eval_steps"],
        save_total_limit=config["training"]["save_total_limit"],
        fp16=config["training"]["fp16"],
        gradient_checkpointing=config["training"]["gradient_checkpointing"],
        evaluation_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        deepspeed=config["configs/deepspeed_config.json"] if config["deepspeed"]["enabled"] else None,
        report_to=["tensorboard"] if config["logging"]["tensorboard"] else [],
    )
    
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        max_length=config["model"]["max_length"],
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    # Train
    print("Starting training...")
    trainer.train()
    
    # Save final model
    final_model_path = os.path.join(output_dir, "final_model")
    model_wrapper.save_lora_weights(final_model_path)
    
    print(f"Training complete. Model saved to {final_model_path}")
    
    return output_dir


if __name__ == "__main__":
    main()
