#!/usr/bin/env python
"""
CPU-based quick evaluation for small sample sizes
Useful for verifying model works without GPU memory constraints
"""

import os
import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# Force CPU only
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from sklearn.metrics import f1_score, accuracy_score

from src.data.data_processor import DataProcessor
from src.data.emotion_taxonomy import TAXONOMY
from src.models.prompt_template import EmotionPromptBuilder, parse_model_output


def load_test_samples(dataset_name, split="test", max_samples=20):
    """Load limited test samples"""
    if dataset_name == "empathetic":
        processor = DataProcessor()
        all_samples = processor.load_samples(f"./cache/data/{split}_samples.json")
        samples = [s for s in all_samples if s.dataset == "empathetic"]
        return samples[:max_samples]
    
    elif dataset_name == "goemotions":
        with open(f"./data/raw/goemotions/{split}.json") as f:
            data = json.load(f)
        
        label_names = [
            "admiration", "amusement", "anger", "annoyance", "approval",
            "caring", "confusion", "curiosity", "desire", "disappointment",
            "disapproval", "disgust", "embarrassment", "excitement", "fear",
            "gratitude", "grief", "joy", "love", "nervousness", "optimism",
            "pride", "realization", "relief", "remorse", "sadness",
            "surprise", "neutral"
        ]
        
        from src.data.data_processor import DialogueSample
        samples = []
        for i, item in enumerate(data[:max_samples]):
            labels = item.get("labels", [])
            if isinstance(labels, int):
                labels = [labels]
            if not labels:
                continue
            
            primary_label = label_names[labels[0]]
            emotion = TAXONOMY.map_emotion(primary_label, "goemotions")
            if emotion:
                sample = DialogueSample(
                    sample_id=f"go_{split}_{i}",
                    dialogue_history=[],
                    target_utterance=item.get("text", ""),
                    emotion=emotion,
                    emotion_idx=TAXONOMY.get_emotion_idx(emotion),
                    speaker="Speaker",
                    dataset="goemotions",
                )
                samples.append(sample)
        return samples
    
    return []


def evaluate_on_cpu(model, tokenizer, samples):
    """Evaluate on CPU sequentially"""
    model.eval()
    builder = EmotionPromptBuilder(use_retrieval=False)
    
    predictions = []
    ground_truths = []
    
    print(f"Evaluating {len(samples)} samples on CPU...")
    print("(Each sample takes ~10-30 seconds on CPU)\n")
    
    for i, sample in enumerate(samples):
        # Build prompt
        prompt = builder.build_inference_prompt(
            dialogue_history=sample.dialogue_history if sample.dialogue_history else [],
            target_utterance=sample.target_utterance,
        )
        
        # Tokenize
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=256,
        )
        
        # Generate on CPU
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=20,
                do_sample=False,
                num_beams=1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        # Decode
        generated = outputs[0][inputs["input_ids"].shape[1]:]
        decoded = tokenizer.decode(generated, skip_special_tokens=True)
        
        # Parse emotion
        result = parse_model_output(decoded)
        pred_emotion = result["emotion"].lower()
        
        # Validate
        if pred_emotion not in TAXONOMY.emotions:
            for e in TAXONOMY.emotions:
                if e in pred_emotion or pred_emotion in e:
                    pred_emotion = e
                    break
            else:
                pred_emotion = "neutral"
        
        predictions.append(pred_emotion)
        ground_truths.append(sample.emotion)
        
        # Show progress and prediction
        print(f"[{i+1}/{len(samples)}] True: {sample.emotion:15s} | Pred: {pred_emotion:15s} | {'✓' if pred_emotion == sample.emotion else '✗'}")
    
    return predictions, ground_truths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--datasets", nargs="+", default=["empathetic"])
    parser.add_argument("--max_samples", type=int, default=20)
    args = parser.parse_args()
    
    print("="*70)
    print("CPU QUICK EVALUATION (Small Sample Test)")
    print("="*70)
    print(f"Model: {args.model_path}")
    print(f"Samples per dataset: {args.max_samples}")
    print(f"Device: CPU only")
    print("="*70)
    
    # Load model to CPU
    print("\nLoading model to CPU (this takes ~2-3 minutes)...")
    base_model = "Qwen/Qwen2.5-7B-Instruct"
    
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        trust_remote_code=True,
        torch_dtype=torch.float32,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )
    
    # Load LoRA weights
    if os.path.exists(f"{args.model_path}/adapter_config.json"):
        model = PeftModel.from_pretrained(model, args.model_path, torch_device="cpu")
        print("✓ Loaded LoRA weights")
    
    model.eval()
    print(f"✓ Model loaded on CPU")
    
    # Evaluate on each dataset
    all_predictions = []
    all_ground_truths = []
    results = {}
    
    for dataset_name in args.datasets:
        print(f"\n{'='*70}")
        print(f"Dataset: {dataset_name.upper()}")
        print(f"{'='*70}")
        
        samples = load_test_samples(dataset_name, "test", args.max_samples)
        if not samples:
            print(f"No samples found for {dataset_name}")
            continue
        
        print(f"Loaded {len(samples)} samples\n")
        
        # Evaluate
        predictions, ground_truths = evaluate_on_cpu(model, tokenizer, samples)
        
        # Calculate metrics
        weighted_f1 = f1_score(ground_truths, predictions, average="weighted", zero_division=0)
        macro_f1 = f1_score(ground_truths, predictions, average="macro", zero_division=0)
        accuracy = accuracy_score(ground_truths, predictions)
        
        print(f"\n{dataset_name.upper()} Results:")
        print(f"  Samples: {len(samples)}")
        print(f"  Accuracy: {accuracy:.4f} ({int(accuracy*len(samples))}/{len(samples)} correct)")
        print(f"  Weighted F1: {weighted_f1:.4f}")
        print(f"  Macro F1: {macro_f1:.4f}")
        
        results[dataset_name] = {
            "samples": len(samples),
            "weighted_f1": weighted_f1,
            "macro_f1": macro_f1,
            "accuracy": accuracy,
        }
        
        all_predictions.extend(predictions)
        all_ground_truths.extend(ground_truths)
    
    # Overall results
    if len(args.datasets) > 1:
        print(f"\n{'='*70}")
        print("OVERALL RESULTS")
        print(f"{'='*70}")
        overall_acc = accuracy_score(all_ground_truths, all_predictions)
        overall_weighted = f1_score(all_ground_truths, all_predictions, average="weighted", zero_division=0)
        
        print(f"  Total samples: {len(all_predictions)}")
        print(f"  Accuracy: {overall_acc:.4f}")
        print(f"  Weighted F1: {overall_weighted:.4f}")
    
    # Check SOTA target (only meaningful if samples >= 100)
    print(f"\n{'='*70}")
    print("NOTE: This is a quick test with only 20 samples.")
    print("For accurate SOTA verification, evaluate on full test set with GPU.")
    print(f"{'='*70}")
    
    return results


if __name__ == "__main__":
    main()
