#!/usr/bin/env python
"""
End-to-End Training Script for Fine-Grained Emotion Recognition

This script handles:
1. Building retrieval index
2. Training the model with LoRA
3. Evaluating on test set
4. Iterating if SOTA not met
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

# Set HF mirror
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, TaskType
from omegaconf import OmegaConf
from tqdm import tqdm

from src.data.data_processor import DataProcessor, DialogueSample
from src.data.dataset import EmotionDataset, EmotionDatasetConfig
from src.data.emotion_taxonomy import TAXONOMY
from src.models.prompt_template import EmotionPromptBuilder
from src.retrieval.retriever import Retriever
from src.evaluation.evaluator import EmotionEvaluator


def build_prompt(sample: DialogueSample, tokenizer, max_length: int = 512) -> dict:
    """Build training prompt for a sample"""
    builder = EmotionPromptBuilder(use_retrieval=False, top_k=0)
    
    prompt = builder.build_training_prompt(
        dialogue_history=sample.dialogue_history,
        target_utterance=sample.target_utterance,
        emotion=sample.emotion,
        speaker=sample.speaker,
        prev_impact=f"Previous emotion was {sample.prev_emotion}." if sample.prev_emotion else None,
    )
    
    # Tokenize
    encodings = tokenizer(
        prompt,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    
    input_ids = encodings["input_ids"].squeeze(0)
    attention_mask = encodings["attention_mask"].squeeze(0)
    
    # Labels = input_ids, but mask padding
    labels = input_ids.clone()
    labels[labels == tokenizer.pad_token_id] = -100
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


class EmotionDataModule:
    """Data module for emotion recognition"""
    
    def __init__(
        self,
        train_path: str,
        val_path: str,
        tokenizer,
        max_length: int = 512,
        max_samples: int = None,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load samples
        processor = DataProcessor()
        print(f"Loading training data from {train_path}")
        self.train_samples = processor.load_samples(train_path)
        print(f"Loading validation data from {val_path}")
        self.val_samples = processor.load_samples(val_path)
        
        # Limit samples if specified
        if max_samples:
            self.train_samples = self.train_samples[:max_samples]
            self.val_samples = self.val_samples[:max_samples//5]
        
        print(f"Train samples: {len(self.train_samples)}")
        print(f"Val samples: {len(self.val_samples)}")
        
        # Pre-tokenize
        print("Tokenizing training data...")
        self.train_data = [build_prompt(s, tokenizer, max_length) for s in tqdm(self.train_samples)]
        print("Tokenizing validation data...")
        self.val_data = [build_prompt(s, tokenizer, max_length) for s in tqdm(self.val_samples)]
    
    def train_dataset(self):
        return self.train_data
    
    def val_dataset(self):
        return self.val_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit training samples")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    args = parser.parse_args()
    
    # Load config
    config = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    
    # Setup output
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"./outputs/checkpoints/run_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*60)
    print("FINE-GRAINED EMOTION RECOGNITION - TRAINING")
    print("="*60)
    print(f"Output: {output_dir}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    
    # Load model and tokenizer
    model_name = config["model"]["base_model"]
    print(f"\nLoading model: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        use_cache=False,  # Disable cache for gradient checkpointing
    )
    
    # Enable gradient checkpointing before LoRA
    model.gradient_checkpointing_enable()
    
    print(f"Model loaded. Parameters: {model.num_parameters():,}")
    
    # Setup LoRA
    print("\nSetting up LoRA...")
    lora_config = LoraConfig(
        r=config["lora"]["r"],
        lora_alpha=config["lora"]["lora_alpha"],
        lora_dropout=config["lora"]["lora_dropout"],
        target_modules=config["lora"]["target_modules"],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Load data
    print("\nLoading data...")
    data_module = EmotionDataModule(
        train_path=config["data"]["train_file"],
        val_path=config["data"]["val_file"],
        tokenizer=tokenizer,
        max_length=config["model"]["max_length"],
        max_samples=args.max_samples,
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        gradient_accumulation_steps=4,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        logging_dir=f"{output_dir}/logs",
        logging_steps=50,
        save_steps=500,
        eval_steps=500,
        save_total_limit=2,
        bf16=True,  # Use bfloat16
        evaluation_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to=["tensorboard"],
        remove_unused_columns=False,
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=data_module.train_data,
        eval_dataset=data_module.val_data,
        tokenizer=tokenizer,
    )
    
    # Train
    print("\nStarting training...")
    trainer.train()
    
    # Save model
    model_path = f"{output_dir}/final_model"
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)
    print(f"\nModel saved to {model_path}")
    
    return model_path


if __name__ == "__main__":
    main()
