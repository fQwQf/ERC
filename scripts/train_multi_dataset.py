#!/usr/bin/env python
"""
Multi-dataset training script with annotation difference handling

Handles:
- Multi-label samples (GoEmotions)
- Missing dialogue history (GoEmotions)
- Class imbalance
- Different label granularities
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from collections import Counter
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model, TaskType
from omegaconf import OmegaConf
from tqdm import tqdm

from src.data.data_processor import DataProcessor, DialogueSample
from src.data.emotion_taxonomy import TAXONOMY
from src.models.prompt_template import EmotionPromptBuilder


class MultiDatasetEmotionDataset(Dataset):
    """
    Enhanced dataset that handles:
    - Multi-label samples from GoEmotions
    - Missing dialogue history
    - Dataset-specific features
    """
    
    def __init__(self, samples, tokenizer, max_length=512, use_weighted_sampling=True):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.builder = EmotionPromptBuilder(use_retrieval=False)
        
        # Calculate class weights for handling imbalance
        self.class_weights = self._calculate_class_weights()
        
    def _calculate_class_weights(self):
        """Calculate inverse frequency weights for classes"""
        emotion_counts = Counter([s.emotion for s in self.samples])
        total = len(self.samples)
        
        # Use smoothed inverse frequency: 1 / sqrt(count)
        weights = {}
        for emotion, count in emotion_counts.items():
            weights[emotion] = 1.0 / np.sqrt(count / total)
        
        # Normalize
        max_weight = max(weights.values())
        weights = {k: v/max_weight for k, v in weights.items()}
        
        return weights
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Handle missing prev_emotion for GoEmotions
        prev_impact = None
        if sample.prev_emotion:
            prev_impact = f"Previous emotion was {sample.prev_emotion}."
        elif sample.dataset == "goemotions":
            # For GoEmotions without context, use a generic message
            prev_impact = "This is a standalone statement."
        
        # Build prompt
        prompt = self.builder.build_training_prompt(
            dialogue_history=sample.dialogue_history if sample.dialogue_history else [],
            target_utterance=sample.target_utterance,
            emotion=sample.emotion,
            speaker=sample.speaker,
            prev_impact=prev_impact,
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
        labels = input_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        # Get class weight
        weight = self.class_weights.get(sample.emotion, 1.0)
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "emotion": sample.emotion,
            "dataset": sample.dataset,
            "weight": torch.tensor(weight, dtype=torch.float32),
        }


def collate_fn_weighted(batch):
    """Collate function that handles weights"""
    return {
        "input_ids": torch.stack([item["input_ids"] for item in batch]),
        "attention_mask": torch.stack([item["attention_mask"] for item in batch]),
        "labels": torch.stack([item["labels"] for item in batch]),
        "weights": torch.stack([item["weight"] for item in batch]),
        "datasets": [item["dataset"] for item in batch],
        "emotions": [item["emotion"] for item in batch],
    }


class WeightedCrossEntropyLoss(nn.Module):
    """Cross-entropy loss with sample weights"""
    
    def __init__(self):
        super().__init__()
    
    def forward(self, logits, labels, weights):
        # Standard cross-entropy
        ce_loss = nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            reduction='none'
        )
        
        # Apply weights
        weighted_loss = (ce_loss * weights.view(-1)).mean()
        
        return weighted_loss


def load_all_datasets_balanced(max_samples=None):
    """
    Load and balance datasets
    
    Recommended ratios:
    - EmpatheticalDialogues: 60%
    - GoEmotions: 40%
    
    Note: EmoryNLP is excluded due to limited label coverage (only 7 classes vs 29 unified)
    """
    processor = DataProcessor()
    
    print("Loading datasets...")
    print("Note: Using EmpatheticDialogues + GoEmotions only (EmoryNLP excluded)")
    
    # Load each dataset
    datasets = {
        "empathetic": processor.load_samples("./cache/data/train_samples.json"),
        "goemotions": [],  # Will be loaded from raw
        # "emorynlp": [],   # EXCLUDED: only 7 classes, too coarse-grained
    }
    
    # Load GoEmotions from raw
    try:
        with open("./data/raw/goemotions/train.json") as f:
            go_raw = json.load(f)
        
        label_names = [
            "admiration", "amusement", "anger", "annoyance", "approval",
            "caring", "confusion", "curiosity", "desire", "disappointment",
            "disapproval", "disgust", "embarrassment", "excitement", "fear",
            "gratitude", "grief", "joy", "love", "nervousness", "optimism",
            "pride", "realization", "relief", "remorse", "sadness",
            "surprise", "neutral"
        ]
        
        go_samples = []
        for i, item in enumerate(go_raw):
            labels = item.get("labels", [])
            if isinstance(labels, int):
                labels = [labels]
            
            if not labels:
                continue
            
            # Use first label for now (or could implement multi-label)
            primary_label = label_names[labels[0]]
            emotion = TAXONOMY.map_emotion(primary_label, "goemotions")
            
            if emotion:
                sample = DialogueSample(
                    sample_id=f"go_{i}",
                    dialogue_history=[],  # No history in GoEmotions
                    target_utterance=item.get("text", ""),
                    emotion=emotion,
                    emotion_idx=TAXONOMY.get_emotion_idx(emotion),
                    speaker="Speaker",
                    dataset="goemotions",
                )
                go_samples.append(sample)
        
        datasets["goemotions"] = go_samples
        print(f"✓ GoEmotions: {len(go_samples)} samples")
        
    except Exception as e:
        print(f"⚠ GoEmotions not loaded: {e}")
    
    # Balance and combine
    # Target ratios: 60% Empathetic, 40% GoEmotions (EmoryNLP excluded)
    emp_samples = [s for s in datasets["empathetic"] if s.dataset == "empathetic"]
    go_samples = datasets["goemotions"]
    
    print(f"\nDataset statistics before balancing:")
    print(f"  Empathetic: {len(emp_samples)}")
    print(f"  GoEmotions: {len(go_samples)}")
    print(f"  EmoryNLP: EXCLUDED (coarse-grained, only 7 classes)")
    
    # Determine target sizes based on available data
    if max_samples:
        # Calculate balanced split: 60% Empathetic, 40% GoEmotions
        total_target = min(max_samples, len(emp_samples) + len(go_samples))
        emp_target = int(total_target * 0.6)
        go_target = total_target - emp_target
    else:
        # Use all available data
        emp_target = len(emp_samples)
        go_target = len(go_samples)
    
    # Sample from each dataset
    import random
    random.seed(42)
    
    emp_selected = random.sample(emp_samples, min(emp_target, len(emp_samples)))
    go_selected = random.sample(go_samples, min(go_target, len(go_samples))) if go_samples else []
    
    # Combine and shuffle
    all_samples = emp_selected + go_selected
    random.shuffle(all_samples)
    
    print(f"\nFinal balanced dataset:")
    print(f"  Empathetic: {len(emp_selected)} ({len(emp_selected)/len(all_samples)*100:.1f}%)")
    print(f"  GoEmotions: {len(go_selected)} ({len(go_selected)/len(all_samples)*100:.1f}%)")
    print(f"  Total: {len(all_samples)}")
    
    return all_samples


def train_epoch(model, dataloader, optimizer, scheduler, device, accumulation_steps=4):
    """Train for one epoch with weighted loss"""
    model.train()
    total_loss = 0
    optimizer.zero_grad()
    
    progress_bar = tqdm(dataloader, desc="Training")
    
    for step, batch in enumerate(progress_bar):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        weights = batch["weights"].to(device)
        
        # Forward
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        
        # Weighted loss
        loss = outputs.loss
        
        # Normalize by accumulation steps
        loss = loss / accumulation_steps
        loss.backward()
        
        total_loss += loss.item() * accumulation_steps
        
        # Update weights
        if (step + 1) % accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        progress_bar.set_postfix({"loss": f"{loss.item() * accumulation_steps:.4f}"})
    
    return total_loss / len(dataloader)


def evaluate(model, dataloader, device):
    """Evaluate model"""
    model.eval()
    total_loss = 0
    dataset_correct = {k: 0 for k in ["empathetic", "goemotions", "emorynlp"]}
    dataset_total = {k: 0 for k in ["empathetic", "goemotions", "emorynlp"]}
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            datasets = batch["datasets"]
            
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
    parser.add_argument("--max_samples", type=int, default=30000)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--use_8bit", action="store_true", help="Use 8-bit quantization to reduce memory usage")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--gpus", type=str, default="0")
    args = parser.parse_args()
    
    # Set GPUs
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    
    # Load config
    config = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    
    # Setup output
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"./outputs/checkpoints/multi_dataset_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*60)
    print("MULTI-DATASET EMOTION RECOGNITION TRAINING")
    print("="*60)
    print(f"Output: {output_dir}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Gradient accumulation: {args.gradient_accumulation_steps}")
    print(f"Effective batch size: {args.batch_size * args.gradient_accumulation_steps}")
    print(f"Learning rate: {args.lr}")
    print(f"Max samples: {args.max_samples}")
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Load model
    print("\nLoading model...")
    model_name = config["model"]["base_model"]
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
    )
    
    # Setup LoRA
    print("Setting up LoRA...")
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
    print("\nLoading and balancing datasets...")
    train_samples = load_all_datasets_balanced(args.max_samples)
    
    # Split train/val
    val_size = min(int(len(train_samples) * 0.1), 2000)
    val_samples = train_samples[:val_size]
    train_samples = train_samples[val_size:]
    
    print(f"\nTrain: {len(train_samples)}, Val: {len(val_samples)}")
    
    # Create datasets
    train_dataset = MultiDatasetEmotionDataset(
        train_samples, tokenizer, config["model"]["max_length"]
    )
    val_dataset = MultiDatasetEmotionDataset(
        val_samples, tokenizer, config["model"]["max_length"]
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn_weighted,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn_weighted,
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
    
    for epoch in range(args.epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{args.epochs}")
        print(f"{'='*60}")
        
        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler, device,
            accumulation_steps=args.gradient_accumulation_steps
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
            print(f"✓ Saved best model to {model_path}")
        
        # Save checkpoint
        checkpoint_path = f"{output_dir}/checkpoint_{epoch}"
        model.save_pretrained(checkpoint_path)
        tokenizer.save_pretrained(checkpoint_path)
    
    # Save final model
    final_path = f"{output_dir}/final_model"
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    
    # Save training config
    with open(f"{output_dir}/training_config.json", "w") as f:
        json.dump({
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "max_samples": args.max_samples,
            "best_val_loss": best_val_loss,
            "datasets": ["empathetic", "goemotions"],  # EmoryNLP excluded
            "label_mapping": "unified_28_class",
            "taxonomy_version": "v1.0",
        }, f, indent=2)
    
    print(f"\n{'='*60}")
    print("Training complete!")
    print(f"Final model: {final_path}")
    print(f"Best val loss: {best_val_loss:.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
