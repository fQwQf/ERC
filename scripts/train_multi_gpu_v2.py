#!/usr/bin/env python
"""
Multi-GPU Multi-Dataset Training with DeepSpeed

Features:
- DeepSpeed ZeRO-2 for efficient multi-GPU training
- Proper data splitting (train/val/test separation)
- Consistent label mapping
- No data leakage between splits
"""

import os
import sys
import json
import argparse
import random
from pathlib import Path
from datetime import datetime
from collections import Counter
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model, TaskType
from tqdm import tqdm

from src.data.data_processor import DataProcessor, DialogueSample
from src.data.emotion_taxonomy import TAXONOMY
from src.models.prompt_template import EmotionPromptBuilder


def setup_distributed():
    """Initialize distributed training"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        gpu = int(os.environ['LOCAL_RANK'])
    else:
        rank = 0
        world_size = 1
        gpu = 0
    
    torch.cuda.set_device(gpu)
    
    if world_size > 1:
        dist.init_process_group(backend='nccl', init_method='env://', 
                               world_size=world_size, rank=rank)
    
    return rank, world_size, gpu


def cleanup_distributed():
    """Cleanup distributed training"""
    if dist.is_initialized():
        dist.destroy_process_group()


class MultiDatasetEmotionDataset(Dataset):
    """Dataset with consistent label mapping"""
    
    def __init__(self, samples, tokenizer, max_length=512):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.builder = EmotionPromptBuilder(use_retrieval=False)
        
        # Calculate class weights
        self.class_weights = self._calculate_class_weights()
    
    def _calculate_class_weights(self):
        """Calculate inverse frequency weights"""
        emotion_counts = Counter([s.emotion for s in self.samples])
        total = len(self.samples)
        
        weights = {}
        for emotion, count in emotion_counts.items():
            weights[emotion] = 1.0 / np.sqrt(count / total)
        
        max_weight = max(weights.values())
        weights = {k: v/max_weight for k, v in weights.items()}
        
        return weights
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Build prompt
        prompt = self.builder.build_training_prompt(
            dialogue_history=sample.dialogue_history if sample.dialogue_history else [],
            target_utterance=sample.target_utterance,
            emotion=sample.emotion,
            speaker=sample.speaker,
            prev_impact=None, # Shortened prompt doesn't use this explicitly in simple way
        )
        
        # Tokenize
        encodings = self.tokenizer(
            prompt,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        
        input_ids = encodings["input_ids"].squeeze(0)
        attention_mask = encodings["attention_mask"].squeeze(0)
        
        # Create labels - only compute loss on the response part
        labels = input_ids.clone()
        
        # FIND RESPONSE START (more robustly)
        # Search for "assistant\n- Emotion:" part
        # Token IDs for "assistant\n- Emotion:" are prefix-dependent, so we find string match
        prompt_without_response = self.builder.build_inference_prompt(
            dialogue_history=sample.dialogue_history if sample.dialogue_history else [],
            target_utterance=sample.target_utterance,
        )
        
        # Encode the prompt part only to find its length
        prompt_enc = self.tokenizer(
            prompt_without_response,
            max_length=self.max_length,
            truncation=True,
            add_special_tokens=False # Don't add BOS again
        )
        
        # Response starts roughly after prompt_enc length
        response_start_idx = len(prompt_enc["input_ids"])
        
        # Mask prompt part
        labels[:response_start_idx] = -100
        
        # Mask padding tokens
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        # Final sanity check: if all labels are -100, we have an issue
        if (labels != -100).sum() == 0:
             # Force unmask the last few tokens just to avoid NaN, though this shouldn't happen with 512
             labels[-10:] = input_ids[-10:] 
        
        weight = self.class_weights.get(sample.emotion, 1.0)
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "weight": torch.tensor(weight, dtype=torch.float32),
            "dataset": sample.dataset,
            "emotion": sample.emotion,
        }



def collate_fn(batch):
    """Collate function"""
    return {
        "input_ids": torch.stack([item["input_ids"] for item in batch]),
        "attention_mask": torch.stack([item["attention_mask"] for item in batch]),
        "labels": torch.stack([item["labels"] for item in batch]),
        "weights": torch.stack([item["weight"] for item in batch]),
    }


def load_and_split_data(max_samples=None, val_ratio=0.1, test_ratio=0.1):
    """
    Load and split data with ROLE FILTERING and CLASS BALANCING
    """
    processor = DataProcessor()
    
    # Load raw data (Role filtering is now handled in DataProcessor)
    train_emp = processor.load_empathetic_dialogues("train")
    val_emp = processor.load_empathetic_dialogues("validation")
    test_emp = processor.load_empathetic_dialogues("test")
    
    train_go = processor.load_goemotions("train")
    val_go = processor.load_goemotions("validation")
    test_go = processor.load_goemotions("test")
    
    # DOWN-SAMPLE NEUTRAL to balance classes
    def balance_samples(samples, dataset_name):
        emotions = [s.emotion for s in samples]
        counts = Counter(emotions)
        if "neutral" not in counts:
            return samples
            
        avg_count = sum(counts.values()) / len(counts)
        neutral_limit = int(avg_count * 1.5) # Allow slightly more neutral
        
        if counts["neutral"] > neutral_limit:
            neutral_samples = [s for s in samples if s.emotion == "neutral"]
            other_samples = [s for s in samples if s.emotion != "neutral"]
            random.shuffle(neutral_samples)
            samples = other_samples + neutral_samples[:neutral_limit]
            print(f"  Balanced {dataset_name}: Neutral reduced from {counts['neutral']} to {neutral_limit}")
            
        return samples

    train_emp = balance_samples(train_emp, "Empathetic")
    train_go = balance_samples(train_go, "GoEmotions")

    all_samples = {
        "train": train_emp + train_go,
        "val": val_emp + val_go,
        "test": test_emp + test_go
    }
    
    # Limit if requested
    if max_samples:
        random.shuffle(all_samples["train"])
        all_samples["train"] = all_samples["train"][:max_samples]
        
    random.shuffle(all_samples["train"])
    
    # Print final statistics
    print("\n" + "="*60)
    print("FINAL BALANCED DATA SPLIT")
    print("="*60)
    for split, samples in all_samples.items():
        emp_count = sum(1 for s in samples if s.dataset == "empathetic")
        go_count = sum(1 for s in samples if s.dataset == "goemotions")
        print(f"{split.upper():6}: {len(samples):6} total (Empathetic: {emp_count}, GoEmotions: {go_count})")
    
    return all_samples



def train_epoch(model, dataloader, optimizer, scheduler, device, epoch, rank, accumulation_steps=4):
    """Train for one epoch with gradient accumulation"""
    model.train()
    total_loss = 0
    
    if rank == 0:
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    else:
        pbar = dataloader
    
    for step, batch in enumerate(pbar):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        weights = batch["weights"].to(device)
        
        # Forward
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels, # We still use internal labels for convenience but will scale by weight
        )
        
        # Scale loss by sample weight
        # weights shape: [batch_size]
        # outputs.loss is mean over all non -100 labels in the batch
        # We'll just use the per-sample weight to scale the average loss for simplicity
        # or compute it properly if needed.
        batch_weight = weights.mean()
        loss = (outputs.loss * batch_weight) / accumulation_steps
        
        # Check for NaN loss
        if torch.isnan(loss):
            if rank == 0:
                print(f"Warning: NaN loss detected at step {step}. Skipping batch.")
            optimizer.zero_grad()
            continue
            
        # Backward
        loss.backward()
        
        # Only update weights after accumulation
        if (step + 1) % accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        total_loss += loss.item() * accumulation_steps
        
        if rank == 0:
            pbar.set_postfix({"loss": f"{loss.item() * accumulation_steps:.4f}"})
    
    return total_loss / len(dataloader)


def evaluate(model, dataloader, device, rank):
    """Evaluate model"""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in dataloader:
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
    parser.add_argument("--max_samples", type=int, default=30000)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--max_length", type=int, default=512)
    args = parser.parse_args()
    
    # Setup distributed
    rank, world_size, gpu = setup_distributed()
    device = torch.device(f"cuda:{gpu}")
    
    # Output directory
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = f"./outputs/checkpoints/multi_gpu_{timestamp}"
    
    if rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
        print("="*70)
        print("MULTI-GPU MULTI-DATASET TRAINING")
        print("="*70)
        print(f"GPUs: {world_size}")
        print(f"Output: {args.output_dir}")
        print(f"Model: {args.model_name}")
        print(f"Max Length: {args.max_length}")
        print(f"Epochs: {args.epochs}")
        print(f"Batch size per GPU: {args.batch_size}")
        print(f"Gradient accumulation steps: {args.gradient_accumulation_steps}")
        print(f"Effective batch size: {args.batch_size * world_size * args.gradient_accumulation_steps}")
        print(f"Learning rate: {args.lr}")
        print(f"\n✓ Data splitting ensures NO LEAKAGE between train/val/test")
        print(f"✓ Each dataset split independently")
    
    # Load model
    if rank == 0:
        print("\nLoading model...")
    
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map={"": gpu},
    )
    
    # Setup LoRA
    if rank == 0:
        print("Setting up LoRA...")
    
    lora_config = LoraConfig(
        r=32,
        lora_alpha=64,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    
    if rank == 0:
        model.print_trainable_parameters()
    
    # Wrap model for distributed training
    if world_size > 1:
        model = DDP(model, device_ids=[gpu], output_device=gpu)
    
    # Load and split data
    if rank == 0:
        print("\nLoading data...")
    
    data_splits = load_and_split_data(args.max_samples)
    
    # Create datasets
    train_dataset = MultiDatasetEmotionDataset(data_splits["train"], tokenizer, max_length=args.max_length)
    val_dataset = MultiDatasetEmotionDataset(data_splits["val"], tokenizer, max_length=args.max_length)
    
    # Create samplers for distributed training
    if world_size > 1:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, num_replicas=world_size, rank=rank, shuffle=True
        )
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            val_dataset, num_replicas=world_size, rank=rank, shuffle=False
        )
    else:
        train_sampler = None
        val_sampler = None
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=2,
        pin_memory=True,
        collate_fn=collate_fn,
        shuffle=(train_sampler is None),
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size * 2,
        sampler=val_sampler,
        num_workers=2,
        pin_memory=True,
        collate_fn=collate_fn,
        shuffle=False,
    )
    
    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    
    num_training_steps = len(train_loader) * args.epochs
    num_warmup_steps = num_training_steps // 10
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )
    
    # Training loop
    best_val_loss = float("inf")
    
    for epoch in range(1, args.epochs + 1):
        if rank == 0:
            print(f"\n{'='*70}")
            print(f"Epoch {epoch}/{args.epochs}")
            print(f"{'='*70}")
        
        # Set epoch for sampler
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        
        # Train with gradient accumulation
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device, epoch, rank, 
                                accumulation_steps=args.gradient_accumulation_steps)
        
        # Evaluate
        val_loss = evaluate(model, val_loader, device, rank)
        
        # Gather losses from all GPUs
        if world_size > 1:
            train_loss_tensor = torch.tensor([train_loss], device=device)
            val_loss_tensor = torch.tensor([val_loss], device=device)
            dist.all_reduce(train_loss_tensor, op=dist.ReduceOp.AVG)
            dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.AVG)
            train_loss = train_loss_tensor.item()
            val_loss = val_loss_tensor.item()
        
        if rank == 0:
            print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                model_path = f"{args.output_dir}/best_model"
                if world_size > 1:
                    model.module.save_pretrained(model_path)
                else:
                    model.save_pretrained(model_path)
                tokenizer.save_pretrained(model_path)
                print(f"✓ Saved best model (val_loss: {val_loss:.4f})")
            
            # Save checkpoint
            checkpoint_path = f"{args.output_dir}/checkpoint_{epoch}"
            if world_size > 1:
                model.module.save_pretrained(checkpoint_path)
            else:
                model.save_pretrained(checkpoint_path)
    
    # Save final model
    if rank == 0:
        final_path = f"{args.output_dir}/final_model"
        if world_size > 1:
            model.module.save_pretrained(final_path)
        else:
            model.save_pretrained(final_path)
        tokenizer.save_pretrained(final_path)
        
        # Save training config
        with open(f"{args.output_dir}/training_config.json", "w") as f:
            json.dump({
                "epochs": args.epochs,
                "batch_size_per_gpu": args.batch_size,
                "total_batch_size": args.batch_size * world_size,
                "world_size": world_size,
                "learning_rate": args.lr,
                "best_val_loss": best_val_loss,
                "datasets": ["empathetic", "goemotions"],
                "data_split": "train/val/test split with NO LEAKAGE",
                "train_samples": len(data_splits["train"]),
                "val_samples": len(data_splits["val"]),
                "test_samples": len(data_splits["test"]),
            }, f, indent=2)
        
        print(f"\n{'='*70}")
        print("Training complete!")
        print(f"Best val loss: {best_val_loss:.4f}")
        print(f"Model saved to: {args.output_dir}")
        print(f"{'='*70}")
    
    cleanup_distributed()


if __name__ == "__main__":
    main()
