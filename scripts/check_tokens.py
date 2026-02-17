from transformers import AutoTokenizer

def check_tokens():
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    s1 = "- Emotion:"
    t1 = tokenizer.encode(s1, add_special_tokens=False)
    print(f"Independent: {t1} -> {tokenizer.convert_ids_to_tokens(t1)}")
    
    s2 = "<|im_start|>assistant\n- Emotion: happiness"
    t2 = tokenizer.encode(s2, add_special_tokens=False)
    print(f"Inside string: {t2} -> {tokenizer.convert_ids_to_tokens(t2)}")
    
    # Try to find t1 in t2
    found = False
    for i in range(len(t2) - len(t1) + 1):
        if t2[i:i+len(t1)] == t1:
            print(f"Found at index {i}")
            found = True
            break
    if not found:
        print("NOT FOUND!")

if __name__ == "__main__":
    check_tokens()
