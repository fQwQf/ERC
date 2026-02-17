#!/usr/bin/env python
import os
import sys
import json
import argparse
import time
from pathlib import Path
from collections import Counter
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from sklearn.metrics import f1_score, accuracy_score

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.data_processor import DataProcessor
from src.data.emotion_taxonomy import TAXONOMY
from src.models.prompt_template import EmotionPromptBuilder, parse_model_output

def evaluate_optimized(model, tokenizer, samples, batch_size=16, device="cuda"):
    model.eval()
    builder = EmotionPromptBuilder(use_retrieval=False)
    
    predictions = []
    ground_truths = []
    
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    for i in tqdm(range(0, len(samples), batch_size), desc="Evaluating"):
        batch = samples[i:i+batch_size]
        prompts = [builder.build_inference_prompt(s.dialogue_history, s.target_utterance) for s in batch]
        
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=768).to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=25,
                do_sample=False,
                num_beams=1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True
            )
        
        input_len = inputs["input_ids"].shape[1]
        decoded_batch = tokenizer.batch_decode(outputs[:, input_len:], skip_special_tokens=True)
        
        for decoded, sample in zip(decoded_batch, batch):
            parsed = parse_model_output(decoded)
            pred_emotion = parsed["emotion"].lower().strip()
            
            # Map to taxonomy
            if pred_emotion not in TAXONOMY.emotions:
                found = False
                for e in TAXONOMY.emotions:
                    if e in pred_emotion or pred_emotion in e:
                        pred_emotion = e
                        found = True
                        break
                if not found:
                    pred_emotion = "neutral"
            
            predictions.append(pred_emotion)
            ground_truths.append(sample.emotion)
            
    return predictions, ground_truths

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="empathetic")
    parser.add_argument("--num_samples", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    
    device = f"cuda:{args.gpu}"
    
    # 1.5B Model is small enough to run in BF16 without quantization on 24GB GPU
    print(f"Loading Qwen2.5-1.5B-Instruct for ZERO-SHOT on {device}...")
    base_model = "Qwen/Qwen2.5-1.5B-Instruct"
    
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map={"": device}
    )
    
    processor = DataProcessor()
    samples = processor.load_samples("cache/data/test_samples.json")
    dataset_samples = [s for s in samples if s.dataset == args.dataset]
    
    if args.num_samples > 0 and args.num_samples < len(dataset_samples):
        dataset_samples = dataset_samples[:args.num_samples]
        
    print(f"Zero-shot evaluation on {len(dataset_samples)} samples from {args.dataset}...")
    
    start_time = time.time()
    preds, gts = evaluate_optimized(model, tokenizer, dataset_samples, batch_size=args.batch_size, device=device)
    elapsed = time.time() - start_time
    
    acc = accuracy_score(gts, preds)
    f1_weighted = f1_score(gts, preds, average="weighted")
    
    print(f"\n{'='*40}")
    print(f"ZERO-SHOT 1.5B RESULTS FOR {args.dataset.upper()}")
    print(f"{'='*40}")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Weighted: {f1_weighted:.4f}")
    
    print("\nPrediction Distribution (Top 10):")
    print(Counter(preds).most_common(10))
    
    # Save results
    output_path = Path(f"outputs/zeroshot_1.5b_{args.dataset}.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({
            "accuracy": acc,
            "f1_weighted": f1_weighted,
            "predictions": Counter(preds),
            "samples": len(preds)
        }, f, indent=2)

if __name__ == "__main__":
    main()
