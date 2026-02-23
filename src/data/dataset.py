"""
PyTorch Dataset for Emotion Recognition with Retrieval Augmentation
"""

import torch
from torch.utils.data import Dataset
from typing import List, Dict, Optional, Any
from dataclasses import dataclass

from transformers import PreTrainedTokenizer

from .data_processor import DialogueSample
from .emotion_taxonomy import TAXONOMY
from ..models.prompt_template import EmotionPromptBuilder, parse_model_output
from ..retrieval.retriever import RetrievedExample


@dataclass
class EmotionDatasetConfig:
    """Configuration for emotion dataset"""
    max_length: int = 512
    use_retrieval: bool = True
    top_k_demonstrations: int = 3
    include_speaker_task: bool = True
    include_impact_task: bool = True


class EmotionDataset(Dataset):
    """PyTorch Dataset for emotion recognition"""
    
    def __init__(
        self,
        samples: List[DialogueSample],
        tokenizer: PreTrainedTokenizer,
        config: EmotionDatasetConfig,
        retrieved_examples: Optional[List[List[RetrievedExample]]] = None,
    ):
        self.samples = samples
        self.tokenizer = tokenizer
        self.config = config
        self.retrieved_examples = retrieved_examples
        self.prompt_builder = EmotionPromptBuilder(
            use_retrieval=config.use_retrieval,
            top_k=config.top_k_demonstrations,
        )
        self.taxonomy = TAXONOMY
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        
        # Get retrieved examples if available
        retrieved = None
        if self.retrieved_examples and idx < len(self.retrieved_examples):
            retrieved = [
                {
                    "dialogue_history": ex.dialogue_history,
                    "target_utterance": ex.target_utterance,
                    "emotion": ex.emotion,
                    "speaker": "Speaker",
                }
                for ex in self.retrieved_examples[idx]
            ]
        
        # Generate prev_impact description
        prev_impact = None
        if self.config.include_impact_task and sample.prev_emotion:
            prev_impact = f"The previous speaker expressed {sample.prev_emotion}, which may have influenced the current emotional state."
        
        # Build prompt
        full_text = self.prompt_builder.build_training_prompt(
            dialogue_history=sample.dialogue_history,
            target_utterance=sample.target_utterance,
            emotion=sample.emotion,
            speaker=sample.speaker,
            prev_impact=prev_impact,
            retrieved_examples=retrieved,
        )
        
        # Tokenize
        encodings = self.tokenizer(
            full_text,
            max_length=self.config.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        
        # For causal LM, labels = input_ids (with padding masked as -100)
        labels = encodings["input_ids"].clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            "input_ids": encodings["input_ids"].squeeze(0),
            "attention_mask": encodings["attention_mask"].squeeze(0),
            "labels": labels.squeeze(0),
            "emotion_idx": torch.tensor(sample.emotion_idx, dtype=torch.long),
            "sample_id": sample.sample_id,
        }


class EmotionDatasetForInference(Dataset):
    """Dataset for inference without labels"""
    
    def __init__(
        self,
        samples: List[DialogueSample],
        tokenizer: PreTrainedTokenizer,
        config: EmotionDatasetConfig,
        retrieved_examples: Optional[List[List[RetrievedExample]]] = None,
    ):
        self.samples = samples
        self.tokenizer = tokenizer
        self.config = config
        self.retrieved_examples = retrieved_examples
        self.prompt_builder = EmotionPromptBuilder(
            use_retrieval=config.use_retrieval,
            top_k=config.top_k_demonstrations,
        )
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        
        # Get retrieved examples if available
        retrieved = None
        if self.retrieved_examples and idx < len(self.retrieved_examples):
            retrieved = [
                {
                    "dialogue_history": ex.dialogue_history,
                    "target_utterance": ex.target_utterance,
                    "emotion": ex.emotion,
                    "speaker": "Speaker",
                }
                for ex in self.retrieved_examples[idx]
            ]
        
        # Build prompt (without target response)
        prompt = self.prompt_builder.build_inference_prompt(
            dialogue_history=sample.dialogue_history,
            target_utterance=sample.target_utterance,
            retrieved_examples=retrieved,
        )
        
        # Tokenize
        encodings = self.tokenizer(
            prompt,
            max_length=self.config.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        
        return {
            "input_ids": encodings["input_ids"].squeeze(0),
            "attention_mask": encodings["attention_mask"].squeeze(0),
            "sample_id": sample.sample_id,
            "true_emotion": sample.emotion,
            "true_emotion_idx": sample.emotion_idx,
            "dialogue_history": sample.dialogue_history,
            "target_utterance": sample.target_utterance,
        }


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Collate function for DataLoader"""
    return {
        "input_ids": torch.stack([item["input_ids"] for item in batch]),
        "attention_mask": torch.stack([item["attention_mask"] for item in batch]),
        "labels": torch.stack([item["labels"] for item in batch]),
        "emotion_idx": torch.stack([item["emotion_idx"] for item in batch]),
        "sample_ids": [item["sample_id"] for item in batch],
    }


def collate_fn_inference(batch: List[Dict]) -> Dict[str, Any]:
    """Collate function for inference DataLoader"""
    return {
        "input_ids": torch.stack([item["input_ids"] for item in batch]),
        "attention_mask": torch.stack([item["attention_mask"] for item in batch]),
        "sample_ids": [item["sample_id"] for item in batch],
        "true_emotions": [item["true_emotion"] for item in batch],
        "true_emotion_idxs": [item["true_emotion_idx"] for item in batch],
        "dialogue_histories": [item["dialogue_history"] for item in batch],
        "target_utterances": [item["target_utterance"] for item in batch],
    }
