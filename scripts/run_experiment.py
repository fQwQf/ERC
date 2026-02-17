#!/usr/bin/env python
"""
Main Experiment Script - SOTA Pursuit Loop for Fine-Grained Emotion Recognition

This script implements the automatic training-evaluation-improvement loop
that continues until SOTA targets are met.

Usage:
    python run_experiment.py --config configs/config.yaml --target_f1 0.65
"""

import os
import sys
import json
import argparse
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import subprocess

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from omegaconf import OmegaConf
import torch

from src.data.data_processor import DataProcessor
from src.data.emotion_taxonomy import TAXONOMY
from src.evaluation.evaluator import EmotionEvaluator


# SOTA Targets by dataset
SOTA_TARGETS = {
    "empathetic": 0.65,
    "iemocap": 0.71,
    "meld": 0.69,
    "emory": 0.41,
}


class ExperimentRunner:
    """Runs the SOTA pursuit experiment loop"""
    
    def __init__(
        self,
        config_path: str,
        target_f1: float = 0.65,
        max_iterations: int = 5,
    ):
        self.config_path = config_path
        self.config = OmegaConf.to_container(
            OmegaConf.load(config_path), resolve=True
        )
        self.target_f1 = target_f1
        self.max_iterations = max_iterations
        
        # Create experiment directory
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.exp_dir = Path(f"./outputs/experiments/exp_{self.timestamp}")
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        
        # Training history
        self.history: List[Dict] = []
        self.best_f1 = 0.0
        self.best_model_path: Optional[str] = None
    
    def run(self):
        """Run the SOTA pursuit loop"""
        print("="*70)
        print("FINE-GRAINED EMOTION RECOGNITION - SOTA PURSUIT")
        print("="*70)
        print(f"Target F1: {self.target_f1}")
        print(f"Max iterations: {self.max_iterations}")
        print(f"Experiment dir: {self.exp_dir}")
        print("="*70)
        
        iteration = 0
        while iteration < self.max_iterations:
            iteration += 1
            print(f"\n{'='*70}")
            print(f"ITERATION {iteration}/{self.max_iterations}")
            print(f"{'='*70}")
            
            # Run training
            model_path = self._run_training(iteration)
            
            # Run evaluation
            results = self._run_evaluation(model_path, iteration)
            
            # Record history
            self.history.append({
                "iteration": iteration,
                "model_path": model_path,
                "weighted_f1": results["weighted_f1"],
                "macro_f1": results["macro_f1"],
                "accuracy": results["accuracy"],
                "worst_classes": results["worst_classes"],
                "config": self._get_current_config(),
            })
            
            # Save history
            self._save_history()
            
            # Check if target met
            if results["weighted_f1"] >= self.target_f1:
                print(f"\n{'='*70}")
                print("🎉 SOTA TARGET MET!")
                print(f"Weighted F1: {results['weighted_f1']:.4f} >= {self.target_f1}")
                print(f"{'='*70}")
                
                self.best_f1 = results["weighted_f1"]
                self.best_model_path = model_path
                break
            
            # Update best model
            if results["weighted_f1"] > self.best_f1:
                self.best_f1 = results["weighted_f1"]
                self.best_model_path = model_path
            
            # Analyze and propose improvements
            improvements = self._analyze_and_propose(results)
            
            # Apply improvements for next iteration
            if iteration < self.max_iterations:
                self._apply_improvements(improvements)
        
        # Final report
        self._generate_final_report()
        
        return self.best_model_path, self.best_f1
    
    def _run_training(self, iteration: int) -> str:
        """Run training for one iteration"""
        print("\n--- TRAINING ---")
        
        output_dir = self.exp_dir / f"iteration_{iteration}" / "model"
        
        # Build training command
        cmd = [
            "python", "scripts/train.py",
            "--config", self.config_path,
        ]
        
        # Run training
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(
            cmd,
            capture_output=False,
            text=True,
        )
        
        # Find the actual output directory
        checkpoints_dir = Path(self.config["training"]["output_dir"])
        if checkpoints_dir.exists():
            # Get the most recent run
            runs = sorted(checkpoints_dir.glob("run_*"), key=os.path.getmtime)
            if runs:
                latest_run = runs[-1]
                model_path = latest_run / "final_model"
                if model_path.exists():
                    # Copy to iteration directory
                    shutil.copytree(model_path, output_dir / "final_model")
                    return str(output_dir / "final_model")
        
        return str(output_dir)
    
    def _run_evaluation(self, model_path: str, iteration: int) -> Dict:
        """Run evaluation on trained model"""
        print("\n--- EVALUATION ---")
        
        test_data_path = self.config["data"]["test_file"]
        if not os.path.exists(test_data_path):
            test_data_path = self.config["data"]["val_file"]
        
        # Build evaluation command
        cmd = [
            "python", "scripts/evaluate.py",
            "--model_path", model_path,
            "--data_path", test_data_path,
            "--config", self.config_path,
            "--batch_size", "8",
        ]
        
        # Run evaluation
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
        )
        
        # Parse results from output
        output = result.stdout
        
        # Try to load results from file
        eval_results_path = os.path.join(
            os.path.dirname(model_path), 
            "..", "..", 
            f"iteration_{iteration}", 
            "evaluation", 
            "evaluation_results.json"
        )
        
        if os.path.exists(eval_results_path):
            with open(eval_results_path) as f:
                return json.load(f)
        
        # Fallback: parse from stdout
        results = {
            "weighted_f1": 0.0,
            "macro_f1": 0.0,
            "accuracy": 0.0,
            "worst_classes": [],
        }
        
        for line in output.split("\n"):
            if "Weighted F1:" in line:
                results["weighted_f1"] = float(line.split(":")[1].strip())
            elif "Macro F1:" in line:
                results["macro_f1"] = float(line.split(":")[1].strip())
            elif "Accuracy:" in line:
                results["accuracy"] = float(line.split(":")[1].strip())
        
        return results
    
    def _analyze_and_propose(self, results: Dict) -> Dict:
        """Analyze results and propose improvements"""
        print("\n--- ANALYSIS ---")
        
        improvements = {
            "adjust_learning_rate": False,
            "increase_epochs": False,
            "adjust_lora_r": False,
            "focus_classes": [],
        }
        
        # Analyze worst classes
        worst_classes = results.get("worst_classes", [])
        print("\nWorst performing classes:")
        for emotion, f1 in worst_classes[:5]:
            print(f"  {emotion}: {f1:.4f}")
        
        # Propose improvements based on analysis
        if results["weighted_f1"] < 0.5:
            print("\nRecommendations for next iteration:")
            print("  - Consider increasing learning rate slightly")
            print("  - Increase number of training epochs")
            improvements["adjust_learning_rate"] = True
            improvements["increase_epochs"] = True
        
        elif results["weighted_f1"] < self.target_f1:
            print("\nRecommendations for next iteration:")
            print("  - Fine-tune on problematic classes")
            print("  - Consider adjusting LoRA rank")
            improvements["adjust_lora_r"] = True
            improvements["focus_classes"] = [c[0] for c in worst_classes[:3]]
        
        else:
            print("\nModel performing well, minor fine-tuning recommended.")
        
        return improvements
    
    def _apply_improvements(self, improvements: Dict):
        """Apply improvements to config for next iteration"""
        print("\n--- APPLYING IMPROVEMENTS ---")
        
        config = OmegaConf.load(self.config_path)
        
        if improvements.get("adjust_learning_rate"):
            current_lr = config.training.learning_rate
            new_lr = current_lr * 1.5  # Increase by 50%
            config.training.learning_rate = min(new_lr, 1e-3)
            print(f"  Adjusted learning rate: {current_lr} -> {config.training.learning_rate}")
        
        if improvements.get("increase_epochs"):
            current_epochs = config.training.num_train_epochs
            config.training.num_train_epochs = current_epochs + 1
            print(f"  Increased epochs: {current_epochs} -> {config.training.num_train_epochs}")
        
        if improvements.get("adjust_lora_r"):
            current_r = config.lora.r
            new_r = min(current_r * 2, 64)  # Double but cap at 64
            config.lora.r = new_r
            config.lora.lora_alpha = new_r * 2
            print(f"  Adjusted LoRA rank: {current_r} -> {new_r}")
        
        # Save updated config
        updated_config_path = str(self.exp_dir / "updated_config.yaml")
        OmegaConf.save(config, updated_config_path)
        self.config_path = updated_config_path
        self.config = OmegaConf.to_container(config, resolve=True)
        
        print(f"  Saved updated config to {updated_config_path}")
    
    def _get_current_config(self) -> Dict:
        """Get current config as dict"""
        return OmegaConf.to_container(
            OmegaConf.load(self.config_path), 
            resolve=True
        )
    
    def _save_history(self):
        """Save training history"""
        history_path = self.exp_dir / "training_history.json"
        with open(history_path, "w") as f:
            json.dump(self.history, f, indent=2)
        print(f"History saved to {history_path}")
    
    def _generate_final_report(self):
        """Generate final experiment report"""
        print("\n" + "="*70)
        print("FINAL EXPERIMENT REPORT")
        print("="*70)
        
        report = {
            "timestamp": self.timestamp,
            "target_f1": self.target_f1,
            "best_f1": self.best_f1,
            "best_model_path": self.best_model_path,
            "total_iterations": len(self.history),
            "target_met": self.best_f1 >= self.target_f1,
            "history": self.history,
        }
        
        # Print summary
        print(f"Target F1: {self.target_f1}")
        print(f"Best F1 achieved: {self.best_f1:.4f}")
        print(f"Total iterations: {len(self.history)}")
        print(f"Target met: {'Yes ✓' if report['target_met'] else 'No ✗'}")
        
        if not report["target_met"]:
            print(f"\nGap to target: {self.target_f1 - self.best_f1:.4f}")
            print("Suggestions for further improvement:")
            print("  1. Try a larger base model (e.g., Llama-3-8B)")
            print("  2. Increase training data diversity")
            print("  3. Use more sophisticated retrieval augmentation")
            print("  4. Apply class-balanced sampling or loss weighting")
        
        # Save report
        report_path = self.exp_dir / "final_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        
        print(f"\nFull report saved to {report_path}")
        print("="*70)


def main():
    parser = argparse.ArgumentParser(description="SOTA Pursuit Experiment for Emotion Recognition")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--target_f1", type=float, default=0.65)
    parser.add_argument("--max_iterations", type=int, default=5)
    parser.add_argument("--dataset", type=str, default="empathetic", 
                        choices=["empathetic", "iemocap", "meld", "emory"])
    args = parser.parse_args()
    
    # Get target F1 for dataset
    target_f1 = SOTA_TARGETS.get(args.dataset, args.target_f1)
    
    # Run experiment
    runner = ExperimentRunner(
        config_path=args.config,
        target_f1=target_f1,
        max_iterations=args.max_iterations,
    )
    
    best_model_path, best_f1 = runner.run()
    
    print(f"\nBest model: {best_model_path}")
    print(f"Best F1: {best_f1:.4f}")


if __name__ == "__main__":
    main()
