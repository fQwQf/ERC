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
from peft import PeftModel
from sklearn.metrics import f1_score, accuracy_score

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.data_processor import DataProcessor, DialogueSample
from src.data.emotion_taxonomy import TAXONOMY
from src.models.prompt_template import EmotionPromptBuilder, parse_model_output
from src.retrieval.retriever import Retriever

def evaluate_rai(model, tokenizer, samples, retriever, batch_size=4, device="cuda"):
    model.eval()
    builder = EmotionPromptBuilder(use_retrieval=True, top_k=1) # Top-1 for speed
    
    predictions = []
    ground_truths = []
    
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    for i in tqdm(range(0, len(samples), batch_size), desc="Evaluating RAI"):
        batch = samples[i:i+batch_size]
        
        prompts = []
        for s in batch:
            # Retrieve demonstrations for each sample
            demos = retriever.retrieve(s, k=1)
            # Convert RetrievedExample to dict for prompt builder
            demo_dicts = [
                {
                    "dialogue_history": d.dialogue_history,
                    "target_utterance": d.target_utterance,
                    "emotion": d.emotion,
                    "speaker": "Speaker", # Simplified
                    "impact": "Contextually similar example"
                }
                for d in demos
            ]
            prompt = builder.build_inference_prompt(s.dialogue_history, s.target_utterance, retrieved_examples=demo_dicts)
            prompts.append(prompt)
        
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=32,
                do_sample=False,
                num_beams=1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        input_len = inputs["input_ids"].shape[1]
        decoded_batch = tokenizer.batch_decode(outputs[:, input_len:], skip_special_tokens=True)
        
        for decoded, sample in zip(decoded_batch, batch):
            parsed = parse_model_output(decoded)
            pred_emotion = parsed["emotion"].lower().strip()
            
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
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    
    device = f"cuda:{args.gpu}"
    
    print(f"Loading model for RAI Evaluation on {device}...")
    base_model = "Qwen/Qwen2.5-7B-Instruct"
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
    )
    
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map={"": device}
    )
    model = PeftModel.from_pretrained(model, args.model_path)
    
    processor = DataProcessor()
    train_samples = processor.load_samples("cache/data/train_samples.json")
    test_samples = processor.load_samples("cache/data/test_samples.json")
    ed_test = [s for s in test_samples if s.dataset == "empathetic"][:args.num_samples]
    
    print("Building Retrieval Index...")
    retriever = Retriever(device=device)
    retriever.build_index(train_samples)
    
    print(f"Evaluating RAI on {len(ed_test)} samples...")
    preds, gts = evaluate_rai(model, tokenizer, ed_test, retriever, batch_size=2, device=device)
    
    acc = accuracy_score(gts, preds)
    f1 = f1_score(gts, preds, average="weighted")
    
    print(f"\nRAI RESULTS:")
    print(f"Accuracy: {acc:.4f}")
    print(f"Weighted F1: {f1:.4f}")

if __name__ == "__main__":
    main()
