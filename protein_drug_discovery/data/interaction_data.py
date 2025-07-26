"""Protein-drug interaction data processing and training dataset creation."""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .protein_data import ProteinDataProcessor
from .drug_data import DrugDataProcessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InteractionDataProcessor:
    """Process protein-drug interaction data for training."""
    
    def __init__(self, cache_dir: str = "data/cache"):
        """Initialize interaction data processor."""
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.protein_processor = ProteinDataProcessor(cache_dir)
        self.drug_processor = DrugDataProcessor(cache_dir)
        self.scaler = StandardScaler()
    
    def create_training_dataset(self, num_samples: int = 1000) -> Dict[str, Any]:
        """
        Create training dataset for protein-drug interaction prediction.
        
        Args:
            num_samples: Number of training samples to generate
            
        Returns:
            Dictionary with training data
        """
        try:
            logger.info(f"Creating training dataset with {num_samples} samples...")
            
            # Get protein and drug data
            proteins = self._get_sample_proteins(num_samples // 10)
            drugs = self._get_sample_drugs(num_samples // 10)
            
            # Generate interaction pairs
            interactions = self._generate_interaction_pairs(proteins, drugs, num_samples)
            
            # Create features and labels
            features, labels = self._create_features_and_labels(interactions)
            
            # Split into train/validation/test
            train_data, test_data = train_test_split(
                list(zip(features, labels)), 
                test_size=0.2, 
                random_state=42
            )
            
            train_data, val_data = train_test_split(
                train_data, 
                test_size=0.25, 
                random_state=42
            )
            
            dataset = {
                "train": {
                    "features": [x[0] for x in train_data],
                    "labels": [x[1] for x in train_data]
                },
                "validation": {
                    "features": [x[0] for x in val_data],
                    "labels": [x[1] for x in val_data]
                },
                "test": {
                    "features": [x[0] for x in test_data],
                    "labels": [x[1] for x in test_data]
                },
                "metadata": {
                    "num_samples": num_samples,
                    "num_proteins": len(proteins),
                    "num_drugs": len(drugs),
                    "feature_dim": len(features[0]) if features else 0
                }
            }
            
            # Cache dataset
            cache_file = self.cache_dir / f"training_dataset_{num_samples}.json"
            with open(cache_file, 'w') as f:
                json.dump(dataset, f, indent=2, default=self._json_serializer)
            
            logger.info(f"Created training dataset: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test")
            return dataset
            
        except Exception as e:
            logger.error(f"Error creating training dataset: {e}")
            return self._create_mock_dataset(num_samples)