#!/usr/bin/env python
"""
Optimized Multi-GPU Training Script for Fine-Grained Emotion Recognition

Uses DataParallel for simple multi-GPU training without DeepSpeed.
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
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model, TaskType
from omegaconf import OmegaConf
from tqdm import tqdm

from src.data.data_processor import DataProcessor
from src.models.prompt_template import EmotionPromptBuilder


class EmotionDataset(Dataset):
    """Simple dataset for emotion recognition"""
    
    def __init__(self, samples, tokenizer, max_length=512):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.builder = EmotionPromptBuilder(use_retrieval=False)
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        prompt = self.builder.build_training_prompt(
            dialogue_history=sample.dialogue_history,
            target_utterance=sample.target_utterance,
            emotion=sample.emotion,
            speaker=sample.speaker,
            prev_impact=f"Previous emotion was {sample.prev_emotion}." if sample.prev_emotion else None,
        )
        
        encodings = self.tokenizer(
            prompt,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        
        input_ids = encodings["input_ids"].squeeze(0)
        attention_mask = encodings["attention_mask"].squeeze(0)
        labels = input_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def train_epoch(model, dataloader, optimizer, scheduler, device, accumulation_steps=4, 
                save_steps=500, output_dir=None, tokenizer=None, global_step=0):
    """Train for one epoch with periodic checkpointing"""
    model.train()
    total_loss = 0
    optimizer.zero_grad()
    
    progress_bar = tqdm(dataloader, desc="Training")
    
    for step, batch in enumerate(progress_bar):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        
        loss = outputs.loss / accumulation_steps
        loss.backward()
        
        total_loss += loss.item() * accumulation_steps
        
        if (step + 1) % accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1
            
            # Save checkpoint every save_steps
            if output_dir and global_step % save_steps == 0:
                checkpoint_path = f"{output_dir}/checkpoint_step_{global_step}"
                model.save_pretrained(checkpoint_path)
                if tokenizer:
                    tokenizer.save_pretrained(checkpoint_path)
                print(f"\nSaved checkpoint to {checkpoint_path}")
        
        progress_bar.set_postfix({"loss": f"{loss.item() * accumulation_steps:.4f}"})
    
    return total_loss / len(dataloader), global_step


def evaluate(model, dataloader, device):
    """Evaluate model"""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            
            total_loss += outputs.loss.item()
    
    return total_loss / len(dataloader)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--gpus", type=str, default="0,1,2,3,4,5,6,7")
    args = parser.parse_args()
    
    # Set GPUs
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    
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
    print(f"GPUs: {args.gpus}")
    
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpus = torch.cuda.device_count()
    print(f"Available GPUs: {n_gpus}")
    
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
        device_map="auto",  # Automatically distribute across GPUs
    )
    
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
    processor = DataProcessor()
    train_samples = processor.load_samples(config["data"]["train_file"])
    val_samples = processor.load_samples(config["data"]["val_file"])
    
    if args.max_samples:
        train_samples = train_samples[:args.max_samples]
        val_samples = val_samples[:args.max_samples//5]
    
    print(f"Train samples: {len(train_samples)}")
    print(f"Val samples: {len(val_samples)}")
    
    # Create datasets
    train_dataset = EmotionDataset(train_samples, tokenizer, config["model"]["max_length"])
    val_dataset = EmotionDataset(val_samples, tokenizer, config["model"]["max_length"])
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    
    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=0.01,
    )
    
    num_training_steps = len(train_loader) * args.epochs
    num_warmup_steps = num_training_steps // 10
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )
    
    # Training loop
    best_val_loss = float("inf")
    global_step = 0
    save_steps = 500  # Save every 500 steps
    
    for epoch in range(args.epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{args.epochs}")
        print(f"{'='*60}")
        
        # Train
        train_loss, global_step = train_epoch(
            model, train_loader, optimizer, scheduler, device,
            save_steps=save_steps, output_dir=output_dir, 
            tokenizer=tokenizer, global_step=global_step
        )
        print(f"Train loss: {train_loss:.4f}")
        
        # Evaluate
        val_loss = evaluate(model, val_loader, device)
        print(f"Val loss: {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_path = f"{output_dir}/best_model"
            model.save_pretrained(model_path)
            tokenizer.save_pretrained(model_path)
            print(f"Saved best model to {model_path}")
        
        # Save checkpoint
        checkpoint_path = f"{output_dir}/checkpoint_{epoch}"
        model.save_pretrained(checkpoint_path)
        tokenizer.save_pretrained(checkpoint_path)
    
    # Save final model
    final_path = f"{output_dir}/final_model"
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"\nFinal model saved to {final_path}")
    
    # Save training config
    with open(f"{output_dir}/training_config.json", "w") as f:
        json.dump({
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "best_val_loss": best_val_loss,
            "train_samples": len(train_samples),
            "val_samples": len(val_samples),
        }, f, indent=2)
    
    return final_path


if __name__ == "__main__":
    main()
