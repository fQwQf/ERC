#!/usr/bin/env python
"""
Multi-dataset evaluation script with consistent label mapping

This script ensures the same label mapping strategy is used for evaluation as for training:
- Maps all dataset labels to unified 28-class taxonomy
- Handles GoEmotions multi-label by using first label
- Handles missing dialogue history
- Reports per-dataset and overall metrics
"""

import os
import sys
import json
import argparse
from pathlib import Path
from collections import Counter, defaultdict
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix

from src.data.data_processor import DataProcessor, DialogueSample
from src.data.emotion_taxonomy import TAXONOMY
from src.models.prompt_template import EmotionPromptBuilder, parse_model_output


class ConsistentLabelMapper:
    """
    Ensures consistent label mapping between training and evaluation
    """
    
    @staticmethod
    def map_empathetic_label(context: str) -> str:
        """
        Extract and map EmpatheticDialogues label from context
        Format: 'emotion: text'
        """
        if not context or ':' not in context:
            return "neutral"
        
        original_emotion = context.split(':')[0].strip().lower()
        mapped = TAXONOMY.map_emotion(original_emotion, "empathetic")
        return mapped if mapped else "neutral"
    
    @staticmethod
    def map_goemotions_label(labels: list, label_names: list) -> str:
        """
        Map GoEmotions label(s) to unified label
        Uses first label (consistent with training)
        """
        if not labels:
            return "neutral"
        
        # Use first label (same as training)
        if isinstance(labels, int):
            labels = [labels]
        
        primary_label = label_names[labels[0]]
        mapped = TAXONOMY.map_emotion(primary_label, "goemotions")
        return mapped if mapped else "neutral"
    
    @staticmethod
    def map_emorynlp_label(emotion: str) -> str:
        """
        Map EmoryNLP label to unified label
        """
        if not emotion:
            return "neutral"
        
        mapped = TAXONOMY.map_emotion(emotion, "emory")
        return mapped if mapped else "neutral"


def load_test_data_with_consistent_mapping(dataset_name: str, split: str = "test"):
    """
    Load test data with consistent label mapping
    
    Args:
        dataset_name: 'empathetic', 'goemotions', or 'emorynlp'
        split: 'train', 'validation', or 'test'
    
    Returns:
        List of DialogueSample with unified labels
    """
    mapper = ConsistentLabelMapper()
    samples = []
    
    if dataset_name == "empathetic":
        # Load from processed cache (already mapped during training)
        processor = DataProcessor()
        all_samples = processor.load_samples(f"./cache/data/{split}_samples.json")
        samples = [s for s in all_samples if s.dataset == "empathetic"]
        
    elif dataset_name == "goemotions":
        # Load from raw and apply consistent mapping
        try:
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
            
            for i, item in enumerate(data):
                labels = item.get("labels", [])
                emotion = mapper.map_goemotions_label(labels, label_names)
                
                sample = DialogueSample(
                    sample_id=f"go_{split}_{i}",
                    dialogue_history=[],  # GoEmotions has no history
                    target_utterance=item.get("text", ""),
                    emotion=emotion,
                    emotion_idx=TAXONOMY.get_emotion_idx(emotion),
                    speaker="Speaker",
                    dataset="goemotions",
                )
                samples.append(sample)
                
        except Exception as e:
            print(f"Error loading GoEmotions {split}: {e}")
    
    elif dataset_name == "emorynlp":
        # Load from raw and apply consistent mapping
        try:
            split_file = {
                "train": "emotion-detection-trn.json",
                "validation": "emotion-detection-dev.json",
                "test": "emotion-detection-tst.json"
            }.get(split, "emotion-detection-tst.json")
            
            with open(f"./data/raw/emorynlp/{split_file}") as f:
                raw_data = json.load(f)
            
            utterance_id = 0
            for ep in raw_data.get("episodes", []):
                for scene in ep.get("scenes", []):
                    for utt in scene.get("utterances", []):
                        emotion = utt.get("emotion")
                        if emotion:
                            unified_emotion = mapper.map_emorynlp_label(emotion)
                            
                            sample = DialogueSample(
                                sample_id=f"em_{split}_{utterance_id}",
                                dialogue_history=[],
                                target_utterance=utt.get("text", ""),
                                emotion=unified_emotion,
                                emotion_idx=TAXONOMY.get_emotion_idx(unified_emotion),
                                speaker=utt.get("speaker", "Speaker"),
                                dataset="emorynlp",
                            )
                            samples.append(sample)
                            utterance_id += 1
                            
        except Exception as e:
            print(f"Error loading EmoryNLP {split}: {e}")
    
    return samples


def evaluate_model(model, tokenizer, samples, batch_size=8, device="cuda"):
    """
    Evaluate model on samples with consistent label mapping
    """
    model.eval()
    builder = EmotionPromptBuilder(use_retrieval=False)
    
    predictions = []
    ground_truths = []
    sample_info = []  # Store dataset source for per-dataset metrics
    
    print(f"Evaluating on {len(samples)} samples...")
    
    with torch.no_grad():
        for i in tqdm(range(0, len(samples), batch_size), desc="Evaluating"):
            batch_samples = samples[i:i+batch_size]
            
            for sample in batch_samples:
                # Build prompt (consistent with training)
                prev_impact = None
                if sample.prev_emotion:
                    prev_impact = f"Previous emotion was {sample.prev_emotion}."
                elif sample.dataset == "goemotions":
                    prev_impact = "This is a standalone statement."
                
                prompt = builder.build_inference_prompt(
                    dialogue_history=sample.dialogue_history if sample.dialogue_history else [],
                    target_utterance=sample.target_utterance,
                )
                
                # Tokenize
                inputs = tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                ).to(device)
                
                # Generate
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
                pred_emotion = result["emotion"].lower()
                
                # Validate emotion is in taxonomy
                if pred_emotion not in TAXONOMY.emotions:
                    # Try to find closest match
                    for e in TAXONOMY.emotions:
                        if e in pred_emotion or pred_emotion in e:
                            pred_emotion = e
                            break
                    else:
                        pred_emotion = "neutral"
                
                predictions.append(pred_emotion)
                ground_truths.append(sample.emotion)
                sample_info.append({
                    "dataset": sample.dataset,
                    "predicted": pred_emotion,
                    "true": sample.emotion,
                })
    
    return predictions, ground_truths, sample_info


def compute_metrics(predictions, ground_truths, sample_info):
    """
    Compute metrics overall and per-dataset
    """
    results = {}
    
    # Overall metrics
    all_labels = sorted(set(predictions + ground_truths))
    label2idx = {l: i for i, l in enumerate(all_labels)}
    
    pred_indices = [label2idx[p] for p in predictions]
    true_indices = [label2idx[g] for g in ground_truths]
    
    results["overall"] = {
        "weighted_f1": f1_score(true_indices, pred_indices, average="weighted"),
        "macro_f1": f1_score(true_indices, pred_indices, average="macro"),
        "accuracy": accuracy_score(true_indices, pred_indices),
        "num_samples": len(predictions),
    }
    
    # Per-dataset metrics
    dataset_samples = defaultdict(lambda: {"preds": [], "trues": []})
    for info in sample_info:
        ds = info["dataset"]
        dataset_samples[ds]["preds"].append(info["predicted"])
        dataset_samples[ds]["trues"].append(info["true"])
    
    for dataset, data in dataset_samples.items():
        preds = data["preds"]
        trues = data["trues"]
        
        # Get labels for this dataset
        ds_labels = sorted(set(preds + trues))
        ds_label2idx = {l: i for i, l in enumerate(ds_labels)}
        
        ds_pred_indices = [ds_label2idx[p] for p in preds]
        ds_true_indices = [ds_label2idx[t] for t in trues]
        
        results[dataset] = {
            "weighted_f1": f1_score(ds_true_indices, ds_pred_indices, average="weighted"),
            "macro_f1": f1_score(ds_true_indices, ds_pred_indices, average="macro"),
            "accuracy": accuracy_score(ds_true_indices, ds_pred_indices),
            "num_samples": len(preds),
        }
        
        # Per-class F1 for this dataset
        per_class = f1_score(ds_true_indices, ds_pred_indices, average=None)
        results[dataset]["per_class_f1"] = {
            ds_labels[i]: float(f1) for i, f1 in enumerate(per_class)
        }
    
    return results


def print_results(results, model_name=""):
    """
    Print evaluation results
    """
    print("\n" + "="*70)
    print(f"EVALUATION RESULTS - {model_name}")
    print("="*70)
    
    # Overall results
    overall = results["overall"]
    print(f"\n【OVERALL】")
    print(f"  Samples: {overall['num_samples']}")
    print(f"  Weighted F1: {overall['weighted_f1']:.4f}")
    print(f"  Macro F1: {overall['macro_f1']:.4f}")
    print(f"  Accuracy: {overall['accuracy']:.4f}")
    
    # SOTA targets
    print(f"\n  SOTA Targets:")
    targets = {
        "empathetic": 0.65,
        "goemotions": None,  # No specific target
    }
    for dataset, target in targets.items():
        if target and dataset in results:
            actual = results[dataset]["weighted_f1"]
            status = "✓ MET" if actual >= target else "✗ NOT MET"
            print(f"    {dataset}: {actual:.4f} / {target:.4f} {status}")
    
    # Per-dataset results
    for dataset in ["empathetic", "goemotions", "emorynlp"]:
        if dataset in results:
            ds_results = results[dataset]
            print(f"\n【{dataset.upper()}】")
            print(f"  Samples: {ds_results['num_samples']}")
            print(f"  Weighted F1: {ds_results['weighted_f1']:.4f}")
            print(f"  Macro F1: {ds_results['macro_f1']:.4f}")
            print(f"  Accuracy: {ds_results['accuracy']:.4f}")
            
            if "per_class_f1" in ds_results:
                print(f"  Top 5 classes by F1:")
                sorted_f1 = sorted(ds_results["per_class_f1"].items(), 
                                  key=lambda x: x[1], reverse=True)[:5]
                for emotion, f1 in sorted_f1:
                    print(f"    {emotion}: {f1:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, 
                       help="Path to trained model")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--datasets", type=str, nargs="+", 
                       default=["empathetic", "goemotions"],
                       choices=["empathetic", "goemotions", "emorynlp", "all"],
                       help="Datasets to evaluate on")
    parser.add_argument("--split", type=str, default="test",
                       choices=["train", "validation", "test"])
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--output", type=str, default=None,
                       help="Output JSON file for results")
    parser.add_argument("--gpus", type=str, default="0")
    args = parser.parse_args()
    
    # Set GPUs
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("="*70)
    print("MULTI-DATASET EVALUATION")
    print("="*70)
    print(f"Model: {args.model_path}")
    print(f"Datasets: {args.datasets}")
    print(f"Split: {args.split}")
    print(f"Device: {device}")
    print(f"\nNote: Using consistent label mapping with training")
    print(f"      - Unified 28-class taxonomy")
    print(f"      - GoEmotions: using first label only")
    print(f"      - EmoryNLP: coarse-grained (7→7 classes)")
    
    # Load model
    print("\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        trust_remote_code=True,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    
    # Load LoRA weights
    model = PeftModel.from_pretrained(model, args.model_path)
    model.eval()
    
    # Evaluate on each dataset
    all_predictions = []
    all_ground_truths = []
    all_sample_info = []
    
    datasets_to_eval = args.datasets
    if "all" in datasets_to_eval:
        datasets_to_eval = ["empathetic", "goemotions", "emorynlp"]
    
    for dataset_name in datasets_to_eval:
        print(f"\n{'='*70}")
        print(f"Evaluating on {dataset_name.upper()}")
        print(f"{'='*70}")
        
        # Load data with consistent mapping
        samples = load_test_data_with_consistent_mapping(dataset_name, args.split)
        
        if args.max_samples:
            samples = samples[:args.max_samples]
        
        print(f"Loaded {len(samples)} samples")
        
        if len(samples) == 0:
            print(f"⚠ No samples found for {dataset_name}, skipping...")
            continue
        
        # Evaluate
        predictions, ground_truths, sample_info = evaluate_model(
            model, tokenizer, samples, args.batch_size, device
        )
        
        all_predictions.extend(predictions)
        all_ground_truths.extend(ground_truths)
        all_sample_info.extend(sample_info)
    
    # Compute metrics
    print("\nComputing metrics...")
    results = compute_metrics(all_predictions, all_ground_truths, all_sample_info)
    
    # Print results
    model_name = Path(args.model_path).name
    print_results(results, model_name)
    
    # Save results
    if args.output:
        output_file = args.output
    else:
        output_dir = Path(args.model_path)
        output_file = output_dir / f"eval_results_{args.split}.json"
    
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to {output_file}")
    
    # Check if we should continue training
    print("\n" + "="*70)
    print("ANALYSIS & RECOMMENDATIONS")
    print("="*70)
    
    empathetic_f1 = results.get("empathetic", {}).get("weighted_f1", 0)
    goemotions_f1 = results.get("goemotions", {}).get("weighted_f1", 0)
    
    if empathetic_f1 >= 0.65:
        print("✅ EmpatheticDialogues SOTA target MET!")
    else:
        print(f"⚠️  EmpatheticDialogues below target: {empathetic_f1:.4f} < 0.65")
        print("   Recommendation: Continue training with more epochs")
    
    if goemotions_f1 > 0:
        print(f"📊 GoEmotions F1: {goemotions_f1:.4f}")
        if goemotions_f1 < 0.3:
            print("   ⚠️  Low cross-dataset generalization")
            print("   Recommendation: Increase GoEmotions ratio in training mix")
        else:
            print("   ✅ Good cross-dataset generalization")


if __name__ == "__main__":
    main()
