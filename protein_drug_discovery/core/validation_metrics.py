"""Validation metrics for protein-drug interaction models.
Implements ROC-AUC, precision, recall, and other evaluation metrics.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from pathlib import Path
import json
from sklearn.metrics import (
    roc_curve, 
    roc_auc_score, 
    precision_recall_curve, 
    average_precision_score,
    precision_score, 
    recall_score, 
    f1_score, 
    confusion_matrix
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ValidationMetrics:
    """Comprehensive validation metrics for protein-drug interaction models."""
    
    def __init__(self, threshold: float = 0.5):
        """Initialize validation metrics calculator.
        
        Args:
            threshold: Classification threshold for binary metrics
        """
        self.threshold = threshold
        self.results = {}
    
    def calculate_metrics(self, 
                         y_true: np.ndarray, 
                         y_pred: np.ndarray, 
                         y_scores: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Calculate comprehensive validation metrics.
        
        Args:
            y_true: Ground truth labels (0/1)
            y_pred: Predicted labels (0/1)
            y_scores: Prediction scores/probabilities (if available)
            
        Returns:
            Dictionary of calculated metrics
        """
        if y_scores is None:
            y_scores = y_pred
            
        metrics = {}
        
        # Basic classification metrics
        metrics['accuracy'] = np.mean(y_true == (y_scores >= self.threshold).astype(int))
        
        # Precision, Recall, F1
        metrics['precision'] = precision_score(y_true, (y_scores >= self.threshold).astype(int), zero_division=0)
        metrics['recall'] = recall_score(y_true, (y_scores >= self.threshold).astype(int), zero_division=0)
        metrics['f1_score'] = f1_score(y_true, (y_scores >= self.threshold).astype(int), zero_division=0)
        
        # ROC-AUC
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, y_scores)
            fpr, tpr, _ = roc_curve(y_true, y_scores)
            metrics['roc_curve'] = {'fpr': fpr.tolist(), 'tpr': tpr.tolist()}
        except Exception as e:
            logger.warning(f"Failed to calculate ROC-AUC: {e}")
            metrics['roc_auc'] = None
        
        # Precision-Recall AUC
        try:
            metrics['pr_auc'] = average_precision_score(y_true, y_scores)
            precision, recall, _ = precision_recall_curve(y_true, y_scores)
            metrics['pr_curve'] = {'precision': precision.tolist(), 'recall': recall.tolist()}
        except Exception as e:
            logger.warning(f"Failed to calculate PR-AUC: {e}")
            metrics['pr_auc'] = None
        
        # Confusion Matrix
        cm = confusion_matrix(y_true, (y_scores >= self.threshold).astype(int))
        metrics['confusion_matrix'] = cm.tolist()
        
        # Calculate TP, FP, TN, FN
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            metrics['true_positives'] = int(tp)
            metrics['false_positives'] = int(fp)
            metrics['true_negatives'] = int(tn)
            metrics['false_negatives'] = int(fn)
            
            # Additional derived metrics
            metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
            metrics['npv'] = tn / (tn + fn) if (tn + fn) > 0 else 0  # Negative Predictive Value
        
        self.results = metrics
        return metrics
    
    def plot_roc_curve(self, save_path: Optional[str] = None) -> None:
        """Plot ROC curve from calculated metrics.
        
        Args:
            save_path: Optional path to save the plot
        """
        if 'roc_curve' not in self.results:
            logger.warning("ROC curve data not available. Run calculate_metrics first.")
            return
        
        fpr = self.results['roc_curve']['fpr']
        tpr = self.results['roc_curve']['tpr']
        roc_auc = self.results['roc_auc']
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                 label=f'ROC curve (area = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"ROC curve saved to {save_path}")
        
        plt.show()
    
    def plot_precision_recall_curve(self, save_path: Optional[str] = None) -> None:
        """Plot Precision-Recall curve from calculated metrics.
        
        Args:
            save_path: Optional path to save the plot
        """
        if 'pr_curve' not in self.results:
            logger.warning("Precision-Recall curve data not available. Run calculate_metrics first.")
            return
        
        precision = self.results['pr_curve']['precision']
        recall = self.results['pr_curve']['recall']
        pr_auc = self.results['pr_auc']
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='green', lw=2, 
                 label=f'PR curve (area = {pr_auc:.3f})')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Precision-Recall curve saved to {save_path}")
        
        plt.show()
    
    def save_metrics(self, save_path: str) -> None:
        """Save calculated metrics to JSON file.
        
        Args:
            save_path: Path to save the metrics JSON file
        """
        if not self.results:
            logger.warning("No metrics to save. Run calculate_metrics first.")
            return
        
        # Create a copy of results without numpy arrays for JSON serialization
        save_results = {}
        for k, v in self.results.items():
            if k not in ['roc_curve', 'pr_curve', 'confusion_matrix']:
                save_results[k] = v
        
        # Add confusion matrix
        if 'confusion_matrix' in self.results:
            save_results['confusion_matrix'] = self.results['confusion_matrix']
        
        # Save to file
        with open(save_path, 'w') as f:
            json.dump(save_results, f, indent=2)
        
        logger.info(f"Metrics saved to {save_path}")
    
    def print_summary(self) -> None:
        """Print a summary of the calculated metrics."""
        if not self.results:
            logger.warning("No metrics to summarize. Run calculate_metrics first.")
            return
        
        print("\n===== Model Validation Metrics =====\n")
        
        if 'accuracy' in self.results:
            print(f"Accuracy:    {self.results['accuracy']:.4f}")
        
        if 'precision' in self.results:
            print(f"Precision:   {self.results['precision']:.4f}")
        
        if 'recall' in self.results:
            print(f"Recall:      {self.results['recall']:.4f}")
        
        if 'f1_score' in self.results:
            print(f"F1 Score:    {self.results['f1_score']:.4f}")
        
        if 'roc_auc' in self.results and self.results['roc_auc'] is not None:
            print(f"ROC-AUC:     {self.results['roc_auc']:.4f}")
        
        if 'pr_auc' in self.results and self.results['pr_auc'] is not None:
            print(f"PR-AUC:      {self.results['pr_auc']:.4f}")
        
        if 'specificity' in self.results:
            print(f"Specificity: {self.results['specificity']:.4f}")
        
        print("\n===== Confusion Matrix =====\n")
        if 'confusion_matrix' in self.results:
            cm = np.array(self.results['confusion_matrix'])
            if cm.shape == (2, 2):
                print(f"TN: {cm[0,0]}\tFP: {cm[0,1]}")
                print(f"FN: {cm[1,0]}\tTP: {cm[1,1]}")
            else:
                print(cm)
        
        print("\n================================\n")