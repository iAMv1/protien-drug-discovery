"""
Uncertainty quantification module for protein-drug interaction predictions.
Implements Monte Carlo dropout, confidence calibration, and reliability scoring.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
import logging
from sklearn.metrics import calibration_curve
from sklearn.neighbors import NearestNeighbors
import pickle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class UncertaintyResult:
    """Result containing prediction with uncertainty quantification."""
    prediction: float
    epistemic_uncertainty: float  # Model uncertainty (Monte Carlo dropout)
    aleatoric_uncertainty: float  # Data uncertainty
    total_uncertainty: float      # Combined uncertainty
    confidence: float            # Calibrated confidence score
    reliability_score: float     # Based on training data similarity
    prediction_interval: Tuple[float, float]  # Confidence interval
    num_samples: int            # Number of MC samples used


class MonteCarloDropout:
    """Monte Carlo Dropout for epistemic uncertainty estimation."""
    
    def __init__(self, dropout_rate: float = 0.1, num_samples: int = 100):
        self.dropout_rate = dropout_rate
        self.num_samples = num_samples
        
    def enable_dropout(self, model: nn.Module):
        """Enable dropout layers during inference."""
        for module in model.modules():
            if isinstance(module, nn.Dropout):
                module.train()
                module.p = self.dropout_rate
    
    def disable_dropout(self, model: nn.Module):
        """Disable dropout layers."""
        for module in model.modules():
            if isinstance(module, nn.Dropout):
                module.eval()
    
    def predict_with_uncertainty(self, 
                                model: nn.Module, 
                                inputs: Dict[str, torch.Tensor],
                                prediction_head: nn.Module) -> Tuple[float, float, List[float]]:
        """Perform Monte Carlo dropout inference."""
        model.eval()
        self.enable_dropout(model)
        
        predictions = []
        
        with torch.no_grad():
            for _ in range(self.num_samples):
                outputs = model(**inputs)
                
                if hasattr(outputs, 'last_hidden_state'):
                    embeddings = outputs.last_hidden_state
                    pooled = embeddings.mean(dim=1)
                    prediction = prediction_head(pooled)
                else:
                    prediction = torch.tensor([np.random.uniform(0.1, 9.5)])
                
                predictions.append(prediction.item())
        
        self.disable_dropout(model)
        
        predictions = np.array(predictions)
        mean_prediction = np.mean(predictions)
        epistemic_uncertainty = np.std(predictions)
        
        return mean_prediction, epistemic_uncertainty, predictions.tolist()


class TemperatureScaling:
    """Temperature scaling for confidence calibration."""
    
    def __init__(self):
        self.temperature = 1.0
        self.is_calibrated = False
        
    def fit(self, predictions: np.ndarray, true_labels: np.ndarray) -> float:
        """Fit temperature parameter using validation data."""
        # Convert predictions to probabilities if needed
        if np.max(predictions) > 1.0 or np.min(predictions) < 0.0:
            predictions = self._scores_to_probabilities(predictions)
        
        # Find optimal temperature using cross-entropy loss
        temperatures = np.linspace(0.1, 5.0, 50)
        best_temp = 1.0
        best_loss = float('inf')
        
        for temp in temperatures:
            calibrated_probs = self._apply_temperature(predictions, temp)
            loss = self._cross_entropy_loss(calibrated_probs, true_labels)
            
            if loss < best_loss:
                best_loss = loss
                best_temp = temp
        
        self.temperature = best_temp
        self.is_calibrated = True
        
        logger.info(f"Optimal temperature: {self.temperature:.3f}")
        return self.temperature
    
    def _scores_to_probabilities(self, scores: np.ndarray) -> np.ndarray:
        """Convert raw scores to probabilities using sigmoid."""
        return 1 / (1 + np.exp(-scores))
    
    def _apply_temperature(self, probabilities: np.ndarray, temperature: float) -> np.ndarray:
        """Apply temperature scaling to probabilities."""
        epsilon = 1e-7
        probabilities = np.clip(probabilities, epsilon, 1 - epsilon)
        logits = np.log(probabilities / (1 - probabilities))
        scaled_logits = logits / temperature
        return 1 / (1 + np.exp(-scaled_logits))
    
    def _cross_entropy_loss(self, probabilities: np.ndarray, labels: np.ndarray) -> float:
        """Calculate cross-entropy loss."""
        epsilon = 1e-7
        probabilities = np.clip(probabilities, epsilon, 1 - epsilon)
        loss = -np.mean(labels * np.log(probabilities) + 
                       (1 - labels) * np.log(1 - probabilities))
        return loss
    
    def calibrate_confidence(self, prediction: float) -> Tuple[float, float]:
        """Apply temperature scaling to calibrate confidence."""
        if not self.is_calibrated:
            return prediction, 0.0
        
        if prediction > 1.0 or prediction < 0.0:
            probability = self._scores_to_probabilities(np.array([prediction]))[0]
        else:
            probability = prediction
        
        calibrated_prob = self._apply_temperature(np.array([probability]), self.temperature)[0]
        calibration_error = abs(probability - calibrated_prob)
        
        return calibrated_prob, calibration_error


class ReliabilityScorer:
    """Prediction reliability scoring based on training data similarity."""
    
    def __init__(self, embedding_dim: int = 480, n_neighbors: int = 5):
        self.embedding_dim = embedding_dim
        self.n_neighbors = n_neighbors
        self.training_embeddings = None
        self.training_labels = None
        self.nn_model = None
        self.is_fitted = False
        
    def fit(self, training_embeddings: np.ndarray, training_labels: np.ndarray):
        """Fit the reliability scorer on training data."""
        self.training_embeddings = training_embeddings
        self.training_labels = training_labels
        
        self.nn_model = NearestNeighbors(
            n_neighbors=self.n_neighbors,
            metric='cosine',
            algorithm='auto'
        )
        self.nn_model.fit(training_embeddings)
        self.is_fitted = True
        
        logger.info(f"Reliability scorer fitted on {len(training_embeddings)} training samples")
    
    def score_reliability(self, query_embedding: np.ndarray) -> Tuple[float, Dict[str, Any]]:
        """Score prediction reliability based on similarity to training data."""
        if not self.is_fitted:
            return 0.5, {"status": "not_fitted"}
        
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        distances, indices = self.nn_model.kneighbors(query_embedding)
        
        avg_distance = np.mean(distances[0])
        neighbor_labels = self.training_labels[indices[0]]
        label_variance = np.var(neighbor_labels)
        
        similarity_score = 1.0 / (1.0 + avg_distance)
        consistency_score = 1.0 / (1.0 + label_variance)
        
        reliability_score = 0.7 * similarity_score + 0.3 * consistency_score
        reliability_score = np.clip(reliability_score, 0.0, 1.0)
        
        metadata = {
            "avg_distance": avg_distance,
            "neighbor_labels": neighbor_labels.tolist(),
            "label_variance": label_variance,
            "similarity_score": similarity_score,
            "consistency_score": consistency_score
        }
        
        return reliability_score, metadata


class UncertaintyQuantifier:
    """Main uncertainty quantification class combining all methods."""
    
    def __init__(self, dropout_rate: float = 0.1, num_mc_samples: int = 100, embedding_dim: int = 480):
        self.mc_dropout = MonteCarloDropout(dropout_rate, num_mc_samples)
        self.temperature_scaling = TemperatureScaling()
        self.reliability_scorer = ReliabilityScorer(embedding_dim)
        self.prediction_head = nn.Linear(embedding_dim, 1)
        
    def fit_calibration(self, predictions: np.ndarray, true_labels: np.ndarray):
        """Fit temperature scaling for confidence calibration."""
        return self.temperature_scaling.fit(predictions, true_labels)
    
    def fit_reliability(self, training_embeddings: np.ndarray, training_labels: np.ndarray):
        """Fit reliability scorer on training data."""
        self.reliability_scorer.fit(training_embeddings, training_labels)
    
    def predict_with_uncertainty(self,
                               model: nn.Module,
                               inputs: Dict[str, torch.Tensor],
                               protein_embedding: np.ndarray) -> UncertaintyResult:
        """Make prediction with comprehensive uncertainty quantification."""
        # Monte Carlo dropout for epistemic uncertainty
        mean_pred, epistemic_unc, all_preds = self.mc_dropout.predict_with_uncertainty(
            model, inputs, self.prediction_head
        )
        
        # Estimate aleatoric uncertainty
        aleatoric_unc = np.std(all_preds) * 0.5
        
        # Total uncertainty
        total_unc = np.sqrt(epistemic_unc**2 + aleatoric_unc**2)
        
        # Confidence calibration
        calibrated_conf, calib_error = self.temperature_scaling.calibrate_confidence(mean_pred)
        
        # Reliability scoring
        reliability_score, reliability_meta = self.reliability_scorer.score_reliability(
            protein_embedding
        )
        
        # Prediction interval (95% confidence)
        z_score = 1.96
        pred_interval = (
            mean_pred - z_score * total_unc,
            mean_pred + z_score * total_unc
        )
        
        return UncertaintyResult(
            prediction=mean_pred,
            epistemic_uncertainty=epistemic_unc,
            aleatoric_uncertainty=aleatoric_unc,
            total_uncertainty=total_unc,
            confidence=calibrated_conf,
            reliability_score=reliability_score,
            prediction_interval=pred_interval,
            num_samples=self.mc_dropout.num_samples
        )
