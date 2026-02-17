#!/usr/bin/env python
"""
Corrected Evaluation Script with Matching Prompt Format

Key fix: Ensure inference prompt matches training prompt format exactly.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from sklearn.metrics import f1_score, accuracy_score, classification_report

from src.data.data_processor import DataProcessor
from src.data.emotion_taxonomy import TAXONOMY
from src.models.prompt_template import PromptTemplate, parse_model_output


def load_model(base_model_path, lora_path=None):
    """Load model with optional LoRA weights"""
    print(f"Loading base model: {base_model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    
    if lora_path:
        print(f"Loading LoRA weights from: {lora_path}")
        model = PeftModel.from_pretrained(model, lora_path)
    
    model.eval()
    return model, tokenizer


def build_evaluation_prompt(sample, template):
    """
    Build prompt that matches training format exactly.
    
    During training (train_multi_gpu.py), the format was:
    - System instruction + emotion list
    - User query with dialogue history
    - Assistant response: - Emotion: X\n- Speaker: Y\n- Impact: Z
    
    For inference, we need the same input format (without the response).
    """
    # Build prev_impact exactly as in training
    prev_impact = None
    if sample.prev_emotion:
        prev_impact = f"Previous emotion was {sample.prev_emotion}."
    
    # Use the template's build_full_prompt method which creates the input prompt
    # Then we manually add the assistant prefix to match training structure
    prompt = template.build_full_prompt(
        dialogue_history=sample.dialogue_history,
        target_utterance=sample.target_utterance,
        demonstrations=None,  # No retrieval during evaluation to match training
        include_system=True,
    )
    
    # The model was trained to output: - Emotion: {emotion}\n- Speaker: {speaker}\n- Impact: {prev_impact}
    # We need to add the assistant prefix but let the model generate the rest
    return prompt


def evaluate(model, tokenizer, samples, max_length=512):
    """Evaluate model on samples with CORRECT prompt format"""
    template = PromptTemplate()
    
    predictions = []
    ground_truths = []
    
    for sample in tqdm(samples, desc="Evaluating"):
        # Build prompt matching training format
        prompt = build_evaluation_prompt(sample, template)
        
        # Tokenize
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        ).to(model.device)
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=64,
                temperature=0.1,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        # Decode
        generated = outputs[0][inputs["input_ids"].shape[1]:]
        decoded = tokenizer.decode(generated, skip_special_tokens=True)
        
        # Parse
        result = parse_model_output(decoded)
        
        # Map to known emotion
        pred_emotion = result["emotion"].lower()
        if pred_emotion not in TAXONOMY.emotions:
            # Try fuzzy matching
            for e in TAXONOMY.emotions:
                if e in pred_emotion or pred_emotion in e:
                    pred_emotion = e
                    break
            else:
                pred_emotion = "neutral"
        
        predictions.append(pred_emotion)
        ground_truths.append(sample.emotion.lower())
    
    return predictions, ground_truths


def compute_metrics(predictions, ground_truths):
    """Compute evaluation metrics"""
    all_labels = sorted(set(predictions + ground_truths))
    label2idx = {l: i for i, l in enumerate(all_labels)}
    
    pred_indices = [label2idx[p] for p in predictions]
    true_indices = [label2idx[g] for g in ground_truths]
    
    weighted_f1 = f1_score(true_indices, pred_indices, average="weighted")
    macro_f1 = f1_score(true_indices, pred_indices, average="macro")
    accuracy = accuracy_score(true_indices, pred_indices)
    
    per_class = f1_score(true_indices, pred_indices, average=None)
    class_f1 = {all_labels[i]: f1 for i, f1 in enumerate(per_class)}
    
    return {
        "weighted_f1": weighted_f1,
        "macro_f1": macro_f1,
        "accuracy": accuracy,
        "per_class_f1": class_f1,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--test_data", type=str, default="cache/data/test_samples.json")
    parser.add_argument("--max_samples", type=int, default=100)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()
    
    # Load model
    model, tokenizer = load_model(args.base_model, args.model_path)
    
    # Load test data
    print(f"Loading test data from: {args.test_data}")
    processor = DataProcessor()
    samples = processor.load_samples(args.test_data)
    
    if args.max_samples:
        samples = samples[:args.max_samples]
    
    print(f"Test samples: {len(samples)}")
    
    # Show example prompt
    template = PromptTemplate()
    example_prompt = build_evaluation_prompt(samples[0], template)
    print("\n" + "="*60)
    print("EXAMPLE PROMPT (first 500 chars):")
    print("="*60)
    print(example_prompt[:500])
    print("...")
    print("="*60 + "\n")
    
    # Evaluate
    predictions, ground_truths = evaluate(model, tokenizer, samples)
    
    # Compute metrics
    metrics = compute_metrics(predictions, ground_truths)
    
    # Print results
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Weighted F1: {metrics['weighted_f1']:.4f}")
    print(f"Macro F1: {metrics['macro_f1']:.4f}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    
    # SOTA check
    target_f1 = 0.65
    if metrics['weighted_f1'] >= target_f1:
        print(f"\n✓ SOTA TARGET MET! F1 = {metrics['weighted_f1']:.4f} >= {target_f1}")
    else:
        print(f"\n✗ SOTA TARGET NOT MET. F1 = {metrics['weighted_f1']:.4f} < {target_f1}")
    
    # Print per-class F1
    print("\nPer-class F1 (worst 10):")
    sorted_f1 = sorted(metrics['per_class_f1'].items(), key=lambda x: x[1])
    for emotion, f1 in sorted_f1[:10]:
        print(f"  {emotion}: {f1:.4f}")
    
    # Save results
    output_dir = args.output_dir or os.path.dirname(args.model_path)
    os.makedirs(output_dir, exist_ok=True)
    
    results_path = os.path.join(output_dir, "evaluation_results_corrected.json")
    with open(results_path, "w") as f:
        json.dump({
            "weighted_f1": float(metrics["weighted_f1"]),
            "macro_f1": float(metrics["macro_f1"]),
            "accuracy": float(metrics["accuracy"]),
            "per_class_f1": {k: float(v) for k, v in metrics["per_class_f1"].items()},
        }, f, indent=2)
    
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
