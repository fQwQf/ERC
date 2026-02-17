#!/usr/bin/env python
"""
Run comprehensive evaluation on multiple datasets
"""

import os
import sys
import json
from pathlib import Path
from tqdm import tqdm

# Set Chinese mirror
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

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
from src.evaluation.evaluator import EmotionEvaluator


def evaluate_model_on_dataset(
    model_path: str,
    test_samples: list,
    config: dict,
    retriever: Retriever = None,
    batch_size: int = 8,
    device: str = "cuda"
) -> dict:
    """Run evaluation on a specific dataset"""
    
    # Load model
    print(f"Loading model from {model_path}")
    use_flash = config["model"].get("use_flash_attention", False)
    model_wrapper = load_model_for_inference(
        base_model=config["model"]["base_model"],
        lora_path=model_path,
        device_map="auto",
        use_flash_attention=use_flash,
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
        print("Retrieving demonstrations...")
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
                generated = output[input_ids.shape[1]:]
                decoded = tokenizer.decode(generated, skip_special_tokens=True)
                
                parsed = parse_model_output(decoded)
                predictions.append(parsed["emotion"])
                ground_truths.append(batch["true_emotions"][i])
                sample_ids.append(batch["sample_ids"][i])
    
    # Evaluate
    evaluator = EmotionEvaluator(output_dir="./outputs/results")
    
    result = evaluator.evaluate(
        predictions=predictions,
        ground_truths=ground_truths,
        sample_ids=sample_ids,
        save_results=True,
    )
    
    return {
        "weighted_f1": result.weighted_f1,
        "macro_f1": result.macro_f1,
        "accuracy": result.accuracy,
        "per_class_f1": result.per_class_f1,
        "total_samples": len(test_samples),
    }


def main():
    # Configuration
    config_path = "configs/config.yaml"
    model_path = "outputs/checkpoints/run_20260211_112814/final_model"
    
    # Load config
    config = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    
    # Check device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device == "cuda":
        print(f"CUDA devices: {torch.cuda.device_count()}")
    
    # Load processor
    processor = DataProcessor()
    
    # Load test data
    test_samples = processor.load_samples("./cache/data/test_samples.json")
    
    # Split by dataset
    datasets = {}
    for sample in test_samples:
        ds = sample.dataset
        if ds not in datasets:
            datasets[ds] = []
        datasets[ds].append(sample)
    
    print("\n" + "="*60)
    print("DATASET DISTRIBUTION")
    print("="*60)
    for ds, samples in datasets.items():
        print(f"{ds}: {len(samples)} samples")
    print("="*60)
    
    # Build retriever
    train_samples = processor.load_samples("./cache/data/train_samples.json")
    retriever = Retriever(
        model_name=config["retrieval"]["embedding_model"],
        cache_dir=config["retrieval"]["cache_dir"],
        device=device,
    )
    
    index_path = os.path.join(config["retrieval"]["cache_dir"], "index")
    if os.path.exists(index_path):
        retriever.load_index(index_path)
    else:
        print("Building retrieval index...")
        retriever.build_index(train_samples)
        retriever.save_index(index_path)
    
    # Evaluate on each dataset
    results = {}
    
    for ds_name, samples in datasets.items():
        print(f"\n{'='*60}")
        print(f"Evaluating on {ds_name.upper()}")
        print(f"{'='*60}")
        print(f"Samples: {len(samples)}")
        
        try:
            result = evaluate_model_on_dataset(
                model_path=model_path,
                test_samples=samples,
                config=config,
                retriever=retriever,
                batch_size=8,
                device=device,
            )
            
            results[ds_name] = result
            
            print(f"\nResults for {ds_name}:")
            print(f"  Weighted F1: {result['weighted_f1']:.4f}")
            print(f"  Macro F1: {result['macro_f1']:.4f}")
            print(f"  Accuracy: {result['accuracy']:.4f}")
            
        except Exception as e:
            print(f"Error evaluating {ds_name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Overall evaluation
    print(f"\n{'='*60}")
    print("OVERALL EVALUATION")
    print(f"{'='*60}")
    
    all_predictions = []
    all_ground_truths = []
    all_sample_ids = []
    
    for ds_name, result in results.items():
        print(f"\n{ds_name.upper()}:")
        print(f"  Weighted F1: {result['weighted_f1']:.4f}")
        print(f"  Macro F1: {result['macro_f1']:.4f}")
        print(f"  Accuracy: {result['accuracy']:.4f}")
        print(f"  Samples: {result['total_samples']}")
    
    # Save results
    output_file = "./outputs/results/multi_dataset_evaluation.json"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Results saved to {output_file}")
    print(f"{'='*60}")
    
    # Check SOTA targets
    print("\n" + "="*60)
    print("SOTA TARGETS COMPARISON")
    print("="*60)
    targets = {
        "empathetic": 0.65,
    }
    
    for ds_name, target in targets.items():
        if ds_name in results:
            actual = results[ds_name]["weighted_f1"]
            status = "✓ MET" if actual >= target else "✗ NOT MET"
            print(f"{ds_name}: {actual:.4f} / {target:.4f} {status}")
    print("="*60)


if __name__ == "__main__":
    main()
