#!/usr/bin/env python
"""
Inference Script for Fine-Grained Emotion Recognition

End-to-end inference with retrieval-augmented demonstrations.

Usage:
    python inference.py --model_path outputs/checkpoints/run_xxx/final_model
    
    # Or interactive mode:
    python inference.py --model_path outputs/checkpoints/run_xxx/final_model --interactive
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from omegaconf import OmegaConf

from src.data.data_processor import DataProcessor
from src.data.emotion_taxonomy import TAXONOMY
from src.models.model import load_model_for_inference, EmotionRecognitionModel
from src.models.prompt_template import EmotionPromptBuilder, parse_model_output
from src.retrieval.retriever import Retriever


class EmotionPredictor:
    """End-to-end emotion prediction with retrieval augmentation"""
    
    def __init__(
        self,
        model_path: str,
        base_model: str = "Qwen/Qwen2.5-7B-Instruct",
        retriever_index_path: Optional[str] = None,
        embedding_model: str = "sentence-transformers/all-mpnet-base-v2",
        top_k_demonstrations: int = 3,
        device: str = "auto",
    ):
        """
        Initialize the emotion predictor.
        
        Args:
            model_path: Path to LoRA weights
            base_model: Base model name or path
            retriever_index_path: Path to FAISS index for retrieval
            embedding_model: Model for computing embeddings
            top_k_demonstrations: Number of demonstrations to retrieve
            device: Device to run on
        """
        print("Loading model...")
        self.model_wrapper = load_model_for_inference(
            base_model=base_model,
            lora_path=model_path,
            device_map=device,
        )
        self.model = self.model_wrapper.get_model()
        self.tokenizer = self.model_wrapper.get_tokenizer()
        self.model.eval()
        
        # Setup retrieval
        self.retriever: Optional[Retriever] = None
        if retriever_index_path and os.path.exists(retriever_index_path):
            print("Loading retrieval index...")
            self.retriever = Retriever(
                model_name=embedding_model,
                device="cuda" if torch.cuda.is_available() else "cpu",
            )
            self.retriever.load_index(retriever_index_path)
        
        self.prompt_builder = EmotionPromptBuilder(
            use_retrieval=self.retriever is not None,
            top_k=top_k_demonstrations,
        )
        self.top_k = top_k_demonstrations
        self.taxonomy = TAXONOMY
    
    def predict(
        self,
        dialogue_history: List[str],
        target_utterance: str,
        max_new_tokens: int = 64,
        temperature: float = 0.1,
    ) -> dict:
        """
        Predict emotion for a target utterance given dialogue history.
        
        Args:
            dialogue_history: List of previous utterances
            target_utterance: The utterance to classify
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Dictionary with predicted emotion, speaker, and impact
        """
        # Retrieve demonstrations if available
        retrieved_examples = None
        if self.retriever:
            retrieved = self.retriever.retrieve_by_text(
                dialogue_history=dialogue_history,
                target_utterance=target_utterance,
                k=self.top_k,
            )
            retrieved_examples = [
                {
                    "dialogue_history": ex.dialogue_history,
                    "target_utterance": ex.target_utterance,
                    "emotion": ex.emotion,
                    "speaker": "Speaker",
                }
                for ex in retrieved
            ]
        
        # Build prompt
        prompt = self.prompt_builder.build_inference_prompt(
            dialogue_history=dialogue_history,
            target_utterance=target_utterance,
            retrieved_examples=retrieved_examples,
        )
        
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode
        generated = outputs[0][inputs["input_ids"].shape[1]:]
        decoded = self.tokenizer.decode(generated, skip_special_tokens=True)
        
        # Parse result
        result = parse_model_output(decoded)
        
        # Add confidence score (simple heuristic based on output format)
        result["confidence"] = self._estimate_confidence(decoded)
        result["raw_output"] = decoded
        
        return result
    
    def _estimate_confidence(self, output: str) -> float:
        """Estimate confidence from output format"""
        # Check if output follows expected format
        has_emotion = "- Emotion:" in output
        has_speaker = "- Speaker:" in output
        
        if has_emotion and has_speaker:
            return 0.9
        elif has_emotion:
            return 0.7
        else:
            return 0.5
    
    def predict_batch(
        self,
        samples: List[dict],
        batch_size: int = 8,
    ) -> List[dict]:
        """
        Predict emotions for a batch of samples.
        
        Args:
            samples: List of dicts with 'dialogue_history' and 'target_utterance'
            batch_size: Batch size for processing
            
        Returns:
            List of prediction results
        """
        results = []
        for i in range(0, len(samples), batch_size):
            batch = samples[i:i+batch_size]
            for sample in batch:
                result = self.predict(
                    dialogue_history=sample.get("dialogue_history", []),
                    target_utterance=sample["target_utterance"],
                )
                results.append(result)
        return results


def interactive_mode(predictor: EmotionPredictor):
    """Run interactive prediction mode"""
    print("\n" + "="*60)
    print("INTERACTIVE EMOTION RECOGNITION")
    print("="*60)
    print("Enter dialogue turns one by one. Type 'predict' to analyze.")
    print("Type 'reset' to start a new conversation, 'quit' to exit.")
    print("="*60 + "\n")
    
    dialogue_history = []
    
    while True:
        try:
            user_input = input("You: ").strip()
            
            if user_input.lower() == "quit":
                print("Goodbye!")
                break
            
            elif user_input.lower() == "reset":
                dialogue_history = []
                print("Conversation reset.\n")
                continue
            
            elif user_input.lower() == "predict":
                if not dialogue_history:
                    print("No dialogue to analyze. Enter some text first.\n")
                    continue
                
                # Use last utterance as target
                target = dialogue_history[-1]
                history = dialogue_history[:-1]
                
                print(f"\nAnalyzing: '{target}'")
                print("-" * 40)
                
                result = predictor.predict(
                    dialogue_history=history,
                    target_utterance=target,
                )
                
                print(f"Predicted Emotion: {result['emotion']}")
                print(f"Speaker: {result.get('speaker', 'Unknown')}")
                if result.get('impact'):
                    print(f"Impact: {result['impact']}")
                print(f"Confidence: {result.get('confidence', 0):.2f}")
                print("-" * 40 + "\n")
                
            else:
                dialogue_history.append(user_input)
                print(f"  [Turn {len(dialogue_history)} recorded]\n")
                
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break


def main():
    parser = argparse.ArgumentParser(description="Emotion Recognition Inference")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to LoRA weights")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--retriever_index", type=str, default=None,
                        help="Path to FAISS retrieval index")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--interactive", action="store_true",
                        help="Run in interactive mode")
    parser.add_argument("--input_file", type=str, default=None,
                        help="Input JSON file with samples to predict")
    parser.add_argument("--output_file", type=str, default=None,
                        help="Output JSON file for predictions")
    args = parser.parse_args()
    
    # Load config if available
    config = {}
    if os.path.exists(args.config):
        config = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    
    # Get retriever path
    retriever_path = args.retriever_index
    if not retriever_path and config:
        retriever_path = os.path.join(
            config.get("retrieval", {}).get("cache_dir", ""),
            "index"
        )
    
    # Initialize predictor
    predictor = EmotionPredictor(
        model_path=args.model_path,
        base_model=args.base_model,
        retriever_index_path=retriever_path,
        embedding_model=config.get("retrieval", {}).get(
            "embedding_model", 
            "sentence-transformers/all-mpnet-base-v2"
        ),
        top_k_demonstrations=config.get("retrieval", {}).get("top_k", 3),
    )
    
    # Run in appropriate mode
    if args.interactive:
        interactive_mode(predictor)
    
    elif args.input_file:
        # Batch prediction from file
        print(f"Loading samples from {args.input_file}")
        with open(args.input_file) as f:
            samples = json.load(f)
        
        print(f"Predicting emotions for {len(samples)} samples...")
        results = predictor.predict_batch(samples)
        
        # Save results
        output_file = args.output_file or args.input_file.replace(".json", "_predictions.json")
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Predictions saved to {output_file}")
    
    else:
        # Demo prediction
        print("\nDemo Prediction:")
        print("-" * 40)
        
        # Example dialogue
        dialogue = [
            "Hey, I heard about what happened. Are you okay?",
            "Not really... I just lost my job yesterday.",
        ]
        target = "I don't know what I'm going to do now."
        
        result = predictor.predict(
            dialogue_history=dialogue,
            target_utterance=target,
        )
        
        print(f"Dialogue: {' '.join(dialogue)}")
        print(f"Target: {target}")
        print()
        print(f"Predicted Emotion: {result['emotion']}")
        print(f"Confidence: {result.get('confidence', 0):.2f}")
        print("-" * 40)


if __name__ == "__main__":
    main()
