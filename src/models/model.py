"""
LoRA Fine-tuned Model for Multi-Task Emotion Recognition
"""

import os
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from peft import (
    LoraConfig,
    get_peft_model,
    PeftModel,
    TaskType,
    prepare_model_for_kbit_training,
)

from ..data.emotion_taxonomy import TAXONOMY


@dataclass
class ModelConfig:
    """Model configuration"""
    base_model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    target_modules: List[str] = None
    use_flash_attention: bool = True
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    
    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = [
                "q_proj", "v_proj", "k_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]


class EmotionRecognitionModel:
    """LoRA fine-tuned model for emotion recognition"""
    
    def __init__(
        self,
        config: ModelConfig,
        device_map: str = "auto",
    ):
        self.config = config
        self.device_map = device_map
        self.taxonomy = TAXONOMY
        
        # Will be loaded
        self.model: Optional[PreTrainedModel] = None
        self.tokenizer: Optional[PreTrainedTokenizer] = None
        self.peft_config: Optional[LoraConfig] = None
    
    def load_base_model(self):
        """Load base model and tokenizer"""
        print(f"Loading base model: {self.config.base_model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.base_model_name,
            trust_remote_code=True,
            padding_side="left",
        )
        
        # Ensure pad token is set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Model loading kwargs
        model_kwargs = {
            "pretrained_model_name_or_path": self.config.base_model_name,
            "trust_remote_code": True,
            "device_map": self.device_map,
        }
        
        # Quantization options
        if self.config.load_in_4bit:
            from transformers import BitsAndBytesConfig
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        elif self.config.load_in_8bit:
            from transformers import BitsAndBytesConfig
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_8bit=True,
            )
        else:
            model_kwargs["torch_dtype"] = torch.float16
        
        # Flash attention
        if self.config.use_flash_attention:
            model_kwargs["attn_implementation"] = "flash_attention_2"
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(**model_kwargs)
        
        print(f"Model loaded. Parameters: {self.model.num_parameters():,}")
    
    def setup_lora(self):
        """Setup LoRA adapter"""
        if self.model is None:
            raise ValueError("Base model not loaded. Call load_base_model() first.")
        
        # Prepare model for training if using quantization
        if self.config.load_in_4bit or self.config.load_in_8bit:
            self.model = prepare_model_for_kbit_training(self.model)
        
        # LoRA configuration
        self.peft_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=self.config.target_modules,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        
        # Apply LoRA
        self.model = get_peft_model(self.model, self.peft_config)
        self.model.print_trainable_parameters()
    
    def load_lora_weights(self, lora_path: str):
        """Load LoRA weights from checkpoint"""
        if self.model is None:
            self.load_base_model()
        
        self.model = PeftModel.from_pretrained(
            self.model,
            lora_path,
            is_trainable=False,
        )
        print(f"LoRA weights loaded from {lora_path}")
    
    def save_lora_weights(self, output_path: str):
        """Save LoRA weights"""
        if self.model is None:
            raise ValueError("Model not loaded")
        
        self.model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)
        print(f"LoRA weights saved to {output_path}")
    
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int = 64,
        temperature: float = 0.1,
        top_p: float = 0.9,
        do_sample: bool = True,
    ) -> str:
        """Generate response"""
        self.model.eval()
        
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode only the new tokens
        new_tokens = outputs[0][input_ids.shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)
    
    def get_model(self) -> PreTrainedModel:
        """Get the underlying model for training"""
        return self.model
    
    def get_tokenizer(self) -> PreTrainedTokenizer:
        """Get the tokenizer"""
        return self.tokenizer


def create_model(
    base_model: str = "Qwen/Qwen2.5-7B-Instruct",
    lora_r: int = 32,
    lora_alpha: int = 64,
    lora_dropout: float = 0.05,
    device_map: str = "auto",
    load_in_4bit: bool = False,
) -> EmotionRecognitionModel:
    """Factory function to create and setup model"""
    config = ModelConfig(
        base_model_name=base_model,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        load_in_4bit=load_in_4bit,
    )
    
    model = EmotionRecognitionModel(config, device_map=device_map)
    model.load_base_model()
    model.setup_lora()
    
    return model


def load_model_for_inference(
    base_model: str = "Qwen/Qwen2.5-7B-Instruct",
    lora_path: str = None,
    device_map: str = "auto",
    use_flash_attention: bool = False,
) -> EmotionRecognitionModel:
    """Load model for inference"""
    config = ModelConfig(base_model_name=base_model, use_flash_attention=use_flash_attention)
    model = EmotionRecognitionModel(config, device_map=device_map)
    model.load_base_model()

    if lora_path:
        model.load_lora_weights(lora_path)

    return model
