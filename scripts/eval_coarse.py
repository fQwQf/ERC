#!/usr/bin/env python
import json
import sys
from pathlib import Path
from sklearn.metrics import f1_score, accuracy_score

# Define mapping from 28 fine-grained to 8 coarse-grained categories
# Based on emotional valence and similarity
VALENCE_MAPPING = {
    # Joyful / Positive
    "joy": "joyful",
    "excitement": "joyful",
    "happiness": "joyful",
    "amusement": "joyful",
    "pride": "joyful",
    "optimism": "joyful",
    "relief": "joyful",
    "hope": "joyful",
    "gratitude": "joyful",
    "love": "joyful",
    
    # Sad / Negative
    "sadness": "sad",
    "loneliness": "sad",
    "grief": "sad",
    "disappointment": "sad",
    "guilt": "sad",
    "shame": "sad",
    
    # Angry / Hostile
    "anger": "angry",
    "frustration": "angry",
    "jealousy": "angry",
    "disgust": "angry",
    
    # Fear / Anxious
    "fear": "anxious",
    "anxiety": "anxious",
    "nervousness": "anxious",
    
    # Surprise
    "surprise": "surprised",
    
    # Thinking / Inquisitive
    "curiosity": "thinking",
    "confusion": "thinking",
    
    # Neutral
    "neutral": "neutral",
    
    # Caring
    "caring": "caring"
}

def evaluate_coarse(results_path):
    with open(results_path, 'r') as f:
        data = json.load(f)
    
    # Note: The original eval script doesn't save raw predictions by default 
    # unless we modified it. Let's check if they exist or if we need to rerun.
    # If not found, I'll inform the user.
    if "predictions" not in data or "ground_truths" not in data:
        print(f"Error: Raw predictions not found in {results_path}")
        return
    
    preds = data["predictions"]
    gts = data["ground_truths"]
    
    coarse_preds = [VALENCE_MAPPING.get(p, "neutral") for p in preds]
    coarse_gts = [VALENCE_MAPPING.get(g, "neutral") for g in gts]
    
    acc = accuracy_score(coarse_gts, coarse_preds)
    f1_w = f1_score(coarse_gts, coarse_preds, average="weighted")
    
    print(f"\n--- COARSE-GRAINED EVALUATION (8 CLASSES) ---")
    print(f"Original Path: {results_path}")
    print(f"Accuracy: {acc:.4f}")
    print(f"Weighted F1: {f1_w:.4f}")
    
    return {"accuracy": acc, "f1_weighted": f1_w}

if __name__ == "__main__":
    # We need raw predictions. Let me check if eval_sota.py saves them.
    pass
