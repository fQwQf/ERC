#!/usr/bin/env python
import os
import sys
import torch
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from rich.live import Live
from rich.text import Text
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.prompt_template import EmotionPromptBuilder, parse_model_output

class EmotionRecognitionTUI:
    def __init__(self, model_path, gpu_id=0):
        self.console = Console()
        self.gpu_id = gpu_id
        self.device = f"cuda:{gpu_id}"
        
        # Environment Configuration for Local/Mirror Access
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        
        self.console.print("[bold blue]Initializing SOTA Emotion Recognition Engine...[/bold blue]")
        
        # Model Loading - Preference: Local Cache -> Mirror
        base_model = "Qwen/Qwen2.5-7B-Instruct"
        local_path = os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct")
        
        model_name_or_path = base_model
        # Check for standard HF cache structure or direct symlink
        if os.path.exists(local_path):
            self.console.print(f"[dim]Detected local cache at {local_path}[/dim]")
            # We use the name but set local_files_only if network fails, 
            # or try to point to the snapshots folder if needed.
            # However, hf_hub handles models--Qwen--Qwen2.5-7B-Instruct automatically if HF_HOME is correct.
            pass

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )
        
        with self.console.status("[bold green]Loading Base Model (7B)...[/bold green]"):
            # Try loading with local_files_only=True first to avoid timeout
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_name_or_path, 
                    trust_remote_code=True, 
                    local_files_only=True
                )
            except:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_name_or_path, 
                    trust_remote_code=True
                )
                
            self.tokenizer.padding_side = "left"
            
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name_or_path,
                    quantization_config=bnb_config,
                    device_map={"": self.device},
                    trust_remote_code=True,
                    local_files_only=True
                )
            except:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name_or_path,
                    quantization_config=bnb_config,
                    device_map={"": self.device},
                    trust_remote_code=True
                )
            
        with self.console.status("[bold green]Loading SOTA LoRA Weights...[/bold green]"):
            self.model = PeftModel.from_pretrained(self.model, model_path)
            self.model.eval()
            
        self.builder = EmotionPromptBuilder(use_retrieval=False) # Simplifed for demo, no local search
        self.console.print("[bold green]✓ System Ready![/bold green]\n")

    def run(self):
        history = []
        
        self.console.print(Panel.fit(
            "Welcome to the [bold magenta]Sisyphus Emotion Recognition TUI[/bold magenta]\n"
            "Analyze fine-grained emotions with SOTA precision (28 classes).\n"
            "Type [bold red]'exit'[/bold red] to quit, [bold yellow]'clear'[/bold yellow] to reset history.",
            title="Emotion AI v1.0",
            border_style="cyan"
        ))

        while True:
            target = Prompt.ask("\n[bold green]Target Utterance[/bold green]")
            
            if target.lower() == 'exit':
                break
            if target.lower() == 'clear':
                history = []
                self.console.print("[italic yellow]History cleared.[/italic yellow]")
                continue

            with self.console.status("[bold cyan]Analyzing...[/bold cyan]"):
                # Build prompt
                prompt = self.builder.build_inference_prompt(history, target)
                
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=64,
                        do_sample=False,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id
                    )
                
                # Decode and parse
                gen_text = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
                analysis = parse_model_output(gen_text)
                
            # Display Results
            self.display_analysis(target, analysis)
            
            # Update history
            history.append(target)
            if len(history) > 5:
                history.pop(0)

    def display_analysis(self, utterance, analysis):
        table = Table(title="Emotional Analysis Report", show_header=True, header_style="bold magenta")
        table.add_column("Field", style="dim", width=12)
        table.add_column("Result", style="bold")
        
        # Color based on valence (simplified)
        color = "white"
        emo = analysis['emotion'].lower()
        if emo in ['joy', 'happiness', 'excitement', 'gratitude', 'love']: color = "green"
        elif emo in ['sadness', 'grief', 'loneliness', 'disappointment']: color = "blue"
        elif emo in ['anger', 'frustration', 'disgust']: color = "red"
        elif emo in ['fear', 'anxiety', 'nervousness']: color = "yellow"
        
        table.add_row("Emotion", f"[{color}]{analysis['emotion'].upper()}[/{color}]")
        table.add_row("Speaker", analysis['speaker'])
        table.add_row("Context Impact", analysis['impact'] or "N/A")
        
        self.console.print(table)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Sisyphus Emotion Recognition TUI")
    parser.add_argument("--model_path", type=str, default="outputs/checkpoints/7b_sota_v2/best_model", 
                        help="Path to local LoRA weights or Hugging Face repo ID")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID to use")
    args = parser.parse_args()
    
    if not os.path.exists(args.model_path) and "/" not in args.model_path:
        print(f"Error: Model not found at {args.model_path}")
        sys.exit(1)
        
    tui = EmotionRecognitionTUI(args.model_path, gpu_id=args.gpu)
    try:
        tui.run()
    except KeyboardInterrupt:
        print("\nExiting...")
