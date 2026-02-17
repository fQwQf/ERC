import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from src.models.prompt_template import EmotionPromptBuilder
from src.data.data_processor import DataProcessor
import os

def inspect():
    model_name = 'Qwen/Qwen2.5-1.5B-Instruct'
    model_path = 'outputs/checkpoints/1.5b_v1/best_model'
    
    print(f"Loading {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map='cuda:1')
    model = PeftModel.from_pretrained(model, model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    builder = EmotionPromptBuilder(use_retrieval=False)
    processor = DataProcessor()
    samples = processor.load_samples('cache/data/test_samples.json')
    ed_samples = [s for s in samples if s.dataset == 'empathetic'][:5]
    go_samples = [s for s in samples if s.dataset == 'goemotions'][:5]
    
    inspect_samples = ed_samples + go_samples
    
    print("\n" + "="*50)
    print("INSPECTING 1.5B FINETUNED MODEL OUTPUTS")
    print("="*50)
    
    for s in inspect_samples:
        print(f"\nDATASET: {s.dataset}")
        prompt = builder.build_inference_prompt(s.dialogue_history, s.target_utterance)
        inputs = tokenizer(prompt, return_tensors='pt').to('cuda:1')
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=50, do_sample=False)
        
        full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        gen_part = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        
        print(f"Target: {s.target_utterance}")
        print(f"True Emotion: {s.emotion}")
        print(f"Generated Raw:\n{gen_part}")
        print("-" * 30)

if __name__ == "__main__":
    inspect()
