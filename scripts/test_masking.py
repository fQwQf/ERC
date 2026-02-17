from transformers import AutoTokenizer
import torch

def test_masking():
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    prompt = "<|im_start|>system\nYou are an expert emotion recognition assistant...<|im_end|>\n<|im_start|>user\nNow analyze this dialogue...\nTarget Utterance: I'm so happy!<|im_end|>\n<|im_start|>assistant\n- Emotion: happiness\n- Speaker: Speaker- Impact: This is a standalone statement.<|im_end|>"
    
    encodings = tokenizer(
        prompt,
        max_length=256,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    
    input_ids = encodings["input_ids"].squeeze(0)
    labels = input_ids.clone()
    
    # Current masking logic
    assistant_start = tokenizer.encode("- Emotion:", add_special_tokens=False)
    print(f"Assistant start tokens: {assistant_start}")
    
    response_start_idx = None
    for i in range(len(input_ids) - len(assistant_start) + 1):
        if input_ids[i:i+len(assistant_start)].tolist() == assistant_start:
            response_start_idx = i
            break
    
    print(f"Response start index: {response_start_idx}")
    
    if response_start_idx is not None:
        labels[:response_start_idx] = -100
        print(f"Masked {response_start_idx} tokens")
        
        # Print what is NOT masked
        unmasked = input_ids[response_start_idx:]
        # Filter out padding (-100 would be padding if we applied it)
        print(f"Unmasked part: '{tokenizer.decode(unmasked, skip_special_tokens=False)}'")
    else:
        print("FAILED TO FIND RESPONSE START")

if __name__ == "__main__":
    test_masking()
