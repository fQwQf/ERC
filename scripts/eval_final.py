import os
import sys
import json
import argparse
from pathlib import Path
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from sklearn.metrics import f1_score, accuracy_score, classification_report

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.data_processor import DataProcessor
from src.data.emotion_taxonomy import TAXONOMY
from src.models.prompt_template import EmotionPromptBuilder, parse_model_output

def evaluate_robust(model, tokenizer, samples, batch_size=4, device="cuda"):
    model.eval()
    builder = EmotionPromptBuilder(use_retrieval=False)
    
    predictions = []
    ground_truths = []
    
    # Set padding side to left for generation
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    for i in tqdm(range(0, len(samples), batch_size), desc="Evaluating"):
        batch = samples[i:i+batch_size]
        prompts = [builder.build_inference_prompt(s.dialogue_history, s.target_utterance) for s in batch]
        
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=32,
                do_sample=False,
                num_beams=1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        for j, output in enumerate(outputs):
            input_len = inputs["input_ids"][j].shape[0]
            generated = output[input_len:]
            decoded = tokenizer.decode(generated, skip_special_tokens=True).strip()
            
            # Use the existing parser
            parsed = parse_model_output(decoded)
            pred_emotion = parsed["emotion"].lower().strip()
            
            # Map to taxonomy if not exactly matching
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
            ground_truths.append(batch[j].emotion)
            
    return predictions, ground_truths

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="empathetic")
    parser.add_argument("--num_samples", type=int, default=500)
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    base_model = "Qwen/Qwen2.5-7B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map={"": device}
    )
    model = PeftModel.from_pretrained(model, args.model_path)
    
    processor = DataProcessor()
    samples = processor.load_samples("cache/data/test_samples.json")
    dataset_samples = [s for s in samples if s.dataset == args.dataset]
    
    if args.num_samples > 0:
        dataset_samples = dataset_samples[:args.num_samples]
        
    print(f"Evaluating {len(dataset_samples)} samples from {args.dataset}...")
    
    preds, gts = evaluate_robust(model, tokenizer, dataset_samples, batch_size=4, device=device)
    
    acc = accuracy_score(gts, preds)
    f1_weighted = f1_score(gts, preds, average="weighted")
    f1_macro = f1_score(gts, preds, average="macro")
    
    print(f"\nResults for {args.dataset}:")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Weighted: {f1_weighted:.4f}")
    print(f"F1 Macro: {f1_macro:.4f}")
    
    # Save detailed report
    report = classification_report(gts, preds, zero_division=0)
    print("\nClassification Report:")
    print(report)

if __name__ == "__main__":
    main()
