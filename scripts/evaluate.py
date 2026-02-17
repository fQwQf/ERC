#!/usr/bin/env python
"""
Evaluation Script for Fine-Grained Emotion Recognition

Usage:
    python evaluate.py --model_path outputs/checkpoints/run_xxx --data_path cache/data/test_samples.json
"""

import os
import sys
import json
import argparse
from pathlib import Path
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from omegaconf import OmegaConf

from src.data.data_processor import DataProcessor, DialogueSample
from src.data.dataset import EmotionDatasetForInference, EmotionDatasetConfig, collate_fn_inference
from src.data.emotion_taxonomy import TAXONOMY
from src.models.model import load_model_for_inference
from src.models.prompt_template import parse_model_output
from src.retrieval.retriever import Retriever, BatchRetriever
from src.evaluation.evaluator import EmotionEvaluator, analyze_errors


def load_test_data(data_path: str) -> list:
    """Load test samples"""
    processor = DataProcessor()
    return processor.load_samples(data_path)


def run_evaluation(
    model_path: str,
    test_samples: list,
    config: dict,
    retriever: Retriever = None,
    batch_size: int = 8,
) -> dict:
    """Run evaluation on test set"""
    
    # Load model
    print(f"Loading model from {model_path}")
    model_wrapper = load_model_for_inference(
        base_model=config["model"]["base_model"],
        lora_path=model_path,
        device_map="auto",
    )
    model = model_wrapper.get_model()
    tokenizer = model_wrapper.get_tokenizer()
    model.eval()
    
    # Create dataset config
    dataset_config = EmotionDatasetConfig(
        max_length=config["model"]["max_length"],
        use_retrieval=config["retrieval"]["top_k"] > 0,
        top_k_demonstrations=config["retrieval"]["top_k"],
    )
    
    # Retrieve demonstrations
    retrieved_examples = None
    if retriever and dataset_config.use_retrieval:
        print("Retrieving demonstrations for test data...")
        batch_retriever = BatchRetriever(retriever, k=dataset_config.top_k_demonstrations)
        retrieved_examples = batch_retriever.retrieve_batch(test_samples)
    
    # Create dataset
    test_dataset = EmotionDatasetForInference(
        samples=test_samples,
        tokenizer=tokenizer,
        config=dataset_config,
        retrieved_examples=retrieved_examples,
    )
    
    # Create dataloader
    dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn_inference,
    )
    
    # Run inference
    predictions = []
    ground_truths = []
    sample_ids = []
    
    print("Running inference...")
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch["input_ids"].cuda()
            attention_mask = batch["attention_mask"].cuda()
            
            # Generate
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=64,
                temperature=0.1,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            
            # Decode outputs
            for i, output in enumerate(outputs):
                # Get only generated part
                generated = output[input_ids.shape[1]:]
                decoded = tokenizer.decode(generated, skip_special_tokens=True)
                
                # Parse prediction
                parsed = parse_model_output(decoded)
                predictions.append(parsed["emotion"])
                ground_truths.append(batch["true_emotions"][i])
                sample_ids.append(batch["sample_ids"][i])
    
    # Evaluate
    evaluator = EmotionEvaluator(
        output_dir=os.path.join(os.path.dirname(model_path), "evaluation")
    )
    
    result = evaluator.evaluate(
        predictions=predictions,
        ground_truths=ground_truths,
        sample_ids=sample_ids,
        save_results=True,
    )
    
    evaluator.print_summary(result)
    
    # Analyze errors
    error_analysis = analyze_errors(result)
    
    return {
        "weighted_f1": result.weighted_f1,
        "macro_f1": result.macro_f1,
        "accuracy": result.accuracy,
        "per_class_f1": result.per_class_f1,
        "worst_classes": evaluator.get_worst_classes(result, k=5),
        "error_analysis": error_analysis,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()
    
    # Load config
    config = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    
    # Load test data
    test_samples = load_test_data(args.data_path)
    print(f"Loaded {len(test_samples)} test samples")
    
    # Build retriever from training data
    train_data_path = config["data"]["train_file"]
    if os.path.exists(train_data_path):
        processor = DataProcessor()
        train_samples = processor.load_samples(train_data_path)
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        retriever = Retriever(
            model_name=config["retrieval"]["embedding_model"],
            cache_dir=config["retrieval"]["cache_dir"],
            device=device,
        )
        
        index_path = os.path.join(config["retrieval"]["cache_dir"], "index")
        if os.path.exists(index_path):
            retriever.load_index(index_path)
        else:
            retriever.build_index(train_samples)
            retriever.save_index(index_path)
    else:
        retriever = None
    
    # Run evaluation
    results = run_evaluation(
        model_path=args.model_path,
        test_samples=test_samples,
        config=config,
        retriever=retriever,
        batch_size=args.batch_size,
    )
    
    # Print final results
    print("\n" + "="*60)
    print("FINAL EVALUATION RESULTS")
    print("="*60)
    print(f"Weighted F1: {results['weighted_f1']:.4f}")
    print(f"Macro F1: {results['macro_f1']:.4f}")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print()
    print("Worst performing classes:")
    for emotion, f1 in results["worst_classes"]:
        print(f"  {emotion}: {f1:.4f}")
    print("="*60)
    
    # Check against target
    target_f1 = config["evaluation"]["target_f1"].get("empathetic", 0.65)
    if results["weighted_f1"] >= target_f1:
        print(f"\n✓ SOTA TARGET MET! F1 = {results['weighted_f1']:.4f} >= {target_f1}")
    else:
        print(f"\n✗ SOTA TARGET NOT MET. F1 = {results['weighted_f1']:.4f} < {target_f1}")
        print("  Suggestions for improvement:")
        for emotion, f1 in results["worst_classes"]:
            print(f"    - Focus on '{emotion}' class (F1 = {f1:.4f})")


if __name__ == "__main__":
    main()
