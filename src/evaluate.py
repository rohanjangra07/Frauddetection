import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score, precision_recall_curve,
    roc_curve, confusion_matrix
)
from typing import Dict, Any, Tuple
from src.utils import logger
from src.config import OUTPUT_DIR

class ModelEvaluator:
    """
    Evaluation suite focused on highly imbalanced fraud detection.
    Prioritizes PR-AUC and Recall over generic accuracy.
    """
    
    def __init__(self, y_true: pd.Series, y_pred_proba: np.ndarray):
        self.y_true = y_true
        self.y_pred_proba = y_pred_proba
        self.optimal_threshold = 0.5
        
    def evaluate_at_threshold(self, threshold: float = 0.5) -> Dict[str, float]:
        """
        Calculates hard-classification metrics at a specific probability threshold.
        """
        y_pred = (self.y_pred_proba >= threshold).astype(int)
        
        return {
            'Precision': precision_score(self.y_true, y_pred, zero_division=0),
            'Recall': recall_score(self.y_true, y_pred, zero_division=0),
            'F1': f1_score(self.y_true, y_pred, zero_division=0),
            'ROC_AUC': roc_auc_score(self.y_true, self.y_pred_proba),
            'PR_AUC': average_precision_score(self.y_true, self.y_pred_proba)
        }

    def optimize_threshold(self, target_metric: str = 'f1') -> float:
        """
        Finds the probability threshold that maximizes F1 score.
        In production, this might be adjusted to prioritize Recall based on business cost.
        """
        logger.info("Optimizing decision threshold...")
        precisions, recalls, thresholds = precision_recall_curve(self.y_true, self.y_pred_proba)
        
        # Calculate F1 for all thresholds
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
        
        optimal_idx = np.argmax(f1_scores)
        self.optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
        
        logger.info(f"Optimal Threshold found at {self.optimal_threshold:.4f} (Max F1: {f1_scores[optimal_idx]:.4f})")
        return self.optimal_threshold

    def plot_pr_curve(self, save_path: str = None):
        """
        Plots the Precision-Recall curve. 
        Highly relevant for imbalanced classes.
        """
        precisions, recalls, _ = precision_recall_curve(self.y_true, self.y_pred_proba)
        pr_auc = average_precision_score(self.y_true, self.y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recalls, precisions, label=f'PR Curve (AUC = {pr_auc:.3f})', color='darkorange')
        plt.xlabel('Recall (Fraud Caught)')
        plt.ylabel('Precision (Prediction Accuracy)')
        plt.title('Precision-Recall Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(OUTPUT_DIR / save_path)
            logger.info(f"PR Curve saved to {save_path}")
        plt.show()

    def plot_f1_vs_threshold(self):
        """
        Visualizes how Precision, Recall, and F1 change across probability thresholds.
        """
        precisions, recalls, thresholds = precision_recall_curve(self.y_true, self.y_pred_proba)
        f1_scores = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-10)
        
        plt.figure(figsize=(10, 6))
        plt.plot(thresholds, precisions[:-1], label='Precision', color='blue')
        plt.plot(thresholds, recalls[:-1], label='Recall', color='red')
        plt.plot(thresholds, f1_scores, label='F1 Score', color='green', linewidth=2)
        plt.axvline(x=self.optimal_threshold, color='black', linestyle='--', label=f'Optimal Thresh ({self.optimal_threshold:.2f})')
        
        plt.xlabel('Probability Threshold')
        plt.ylabel('Score')
        plt.title('Metrics vs. Decision Threshold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

    def plot_confusion_matrix(self, threshold: float = None):
        """
        Plots the confusion matrix using the optimal or provided threshold.
        """
        thresh = threshold or self.optimal_threshold
        y_pred = (self.y_pred_proba >= thresh).astype(int)
        cm = confusion_matrix(self.y_true, y_pred)
        
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Legit', 'Fraud'], yticklabels=['Legit', 'Fraud'])
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title(f'Confusion Matrix (Threshold = {thresh:.3f})')
        plt.show()
