"""
Evaluation Module for Fine-Grained Emotion Recognition
"""

import os
import json
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from collections import Counter

from sklearn.metrics import (
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
)
import matplotlib.pyplot as plt
import seaborn as sns

from ..data.emotion_taxonomy import TAXONOMY


@dataclass
class EvaluationResult:
    """Container for evaluation results"""
    weighted_f1: float
    macro_f1: float
    accuracy: float
    per_class_f1: Dict[str, float]
    confusion_matrix: np.ndarray
    predictions: List[str]
    ground_truths: List[str]
    misclassified_samples: List[Dict]
    
    def to_dict(self) -> Dict:
        return {
            "weighted_f1": self.weighted_f1,
            "macro_f1": self.macro_f1,
            "accuracy": self.accuracy,
            "per_class_f1": self.per_class_f1,
            "confusion_matrix": self.confusion_matrix.tolist(),
        }


class EmotionEvaluator:
    """Evaluator for emotion recognition models"""
    
    def __init__(
        self,
        emotions: List[str] = None,
        output_dir: str = "./outputs/results",
    ):
        self.emotions = emotions or TAXONOMY.emotions
        self.emotion2idx = {e: i for i, e in enumerate(self.emotions)}
        self.idx2emotion = {i: e for i, e in enumerate(self.emotions)}
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def evaluate(
        self,
        predictions: List[str],
        ground_truths: List[str],
        sample_ids: List[str] = None,
        save_results: bool = True,
    ) -> EvaluationResult:
        """Evaluate predictions against ground truths"""
        
        # Convert to indices
        pred_indices = [self.emotion2idx.get(p.lower(), 0) for p in predictions]
        true_indices = [self.emotion2idx.get(g.lower(), 0) for g in ground_truths]
        
        # Calculate metrics
        weighted_f1 = f1_score(true_indices, pred_indices, average="weighted")
        macro_f1 = f1_score(true_indices, pred_indices, average="macro")
        accuracy = accuracy_score(true_indices, pred_indices)
        
        # Per-class F1
        per_class_f1_values = f1_score(
            true_indices, pred_indices, average=None, zero_division=0
        )
        per_class_f1 = {
            self.emotions[i]: float(f1) 
            for i, f1 in enumerate(per_class_f1_values)
        }
        
        # Confusion matrix
        cm = confusion_matrix(true_indices, pred_indices)
        
        # Find misclassified samples
        misclassified = []
        if sample_ids:
            for i, (pred, true) in enumerate(zip(predictions, ground_truths)):
                if pred.lower() != true.lower():
                    misclassified.append({
                        "sample_id": sample_ids[i],
                        "predicted": pred,
                        "ground_truth": true,
                    })
        
        result = EvaluationResult(
            weighted_f1=weighted_f1,
            macro_f1=macro_f1,
            accuracy=accuracy,
            per_class_f1=per_class_f1,
            confusion_matrix=cm,
            predictions=predictions,
            ground_truths=ground_truths,
            misclassified_samples=misclassified,
        )
        
        if save_results:
            self._save_results(result)
            self._plot_confusion_matrix(cm)
            self._plot_per_class_f1(per_class_f1)
        
        return result
    
    def _save_results(self, result: EvaluationResult):
        """Save results to JSON"""
        path = os.path.join(self.output_dir, "evaluation_results.json")
        with open(path, "w") as f:
            json.dump(result.to_dict(), f, indent=2)
        print(f"Results saved to {path}")
    
    def _plot_confusion_matrix(self, cm: np.ndarray):
        """Plot and save confusion matrix"""
        plt.figure(figsize=(14, 12))
        
        # Normalize
        cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        
        sns.heatmap(
            cm_normalized,
            annot=True,
            fmt=".2f",
            cmap="Blues",
            xticklabels=self.emotions,
            yticklabels=self.emotions,
        )
        plt.title("Normalized Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()
        
        path = os.path.join(self.output_dir, "confusion_matrix.png")
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"Confusion matrix saved to {path}")
    
    def _plot_per_class_f1(self, per_class_f1: Dict[str, float]):
        """Plot per-class F1 scores"""
        plt.figure(figsize=(12, 6))
        
        emotions = list(per_class_f1.keys())
        f1_values = list(per_class_f1.values())
        
        # Sort by F1 score
        sorted_pairs = sorted(zip(emotions, f1_values), key=lambda x: x[1])
        emotions, f1_values = zip(*sorted_pairs)
        
        colors = ["red" if f1 < 0.5 else "green" if f1 > 0.7 else "orange" for f1 in f1_values]
        
        plt.barh(emotions, f1_values, color=colors)
        plt.xlabel("F1 Score")
        plt.title("Per-Class F1 Scores")
        plt.xlim(0, 1)
        plt.axvline(x=0.65, color="red", linestyle="--", label="Target (0.65)")
        plt.legend()
        plt.tight_layout()
        
        path = os.path.join(self.output_dir, "per_class_f1.png")
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"Per-class F1 plot saved to {path}")
    
    def get_worst_classes(
        self, 
        result: EvaluationResult, 
        k: int = 5
    ) -> List[Tuple[str, float]]:
        """Get the k classes with lowest F1 scores"""
        sorted_f1 = sorted(
            result.per_class_f1.items(), 
            key=lambda x: x[1]
        )
        return sorted_f1[:k]
    
    def get_class_distribution(
        self, 
        labels: List[str]
    ) -> Dict[str, int]:
        """Get distribution of labels"""
        counter = Counter(labels)
        return dict(counter.most_common())
    
    def print_summary(self, result: EvaluationResult):
        """Print evaluation summary"""
        print("\n" + "="*60)
        print("EVALUATION SUMMARY")
        print("="*60)
        print(f"Weighted F1: {result.weighted_f1:.4f}")
        print(f"Macro F1: {result.macro_f1:.4f}")
        print(f"Accuracy: {result.accuracy:.4f}")
        print()
        
        print("Top 5 Worst Performing Classes:")
        worst = self.get_worst_classes(result, k=5)
        for emotion, f1 in worst:
            print(f"  {emotion}: {f1:.4f}")
        
        print()
        print("Top 5 Best Performing Classes:")
        best = sorted(result.per_class_f1.items(), key=lambda x: -x[1])[:5]
        for emotion, f1 in best:
            print(f"  {emotion}: {f1:.4f}")
        
        print("="*60)


def analyze_errors(
    result: EvaluationResult,
    top_n: int = 10,
) -> Dict:
    """Analyze common error patterns"""
    
    # Count error types
    error_pairs = Counter()
    for sample in result.misclassified_samples:
        pair = (sample["ground_truth"], sample["predicted"])
        error_pairs[pair] += 1
    
    most_common_errors = error_pairs.most_common(top_n)
    
    print("\nMost Common Misclassifications:")
    for (true, pred), count in most_common_errors:
        print(f"  {true} -> {pred}: {count} times")
    
    return {
        "most_common_errors": [
            {"true": t, "predicted": p, "count": c}
            for (t, p), c in most_common_errors
        ],
        "total_errors": len(result.misclassified_samples),
        "error_rate": len(result.misclassified_samples) / len(result.predictions),
    }
