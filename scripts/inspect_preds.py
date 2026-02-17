import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from src.data.data_processor import DataProcessor
from src.models.prompt_template import EmotionPromptBuilder, parse_model_output

def inspect_predictions():
    model_path = "outputs/checkpoints/multi_gpu_20260212_174600/final_model"
    base_model = "Qwen/Qwen2.5-7B-Instruct"
    
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(model, model_path)
    model.eval()
    
    processor = DataProcessor()
    samples = processor.load_samples("cache/data/test_samples.json")
    emp_samples = [s for s in samples if s.dataset == "empathetic"][:10]
    
    builder = EmotionPromptBuilder(use_retrieval=False)
    
    for i, sample in enumerate(emp_samples):
        prompt = builder.build_inference_prompt(
            dialogue_history=sample.dialogue_history,
            target_utterance=sample.target_utterance
        )
        
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=64)
            
        decoded = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        print(f"Sample {i+1}:")
        print(f"Target: {sample.target_utterance}")
        print(f"True Emotion: {sample.emotion}")
        print(f"Model Output:\n{decoded}")
        print("-" * 30)

if __name__ == "__main__":
    inspect_predictions()
