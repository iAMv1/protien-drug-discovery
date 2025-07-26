"""
Enhanced DoubleSG-DTA Training Script
Combining DoubleSG-DTA architecture with ESM-2 protein embeddings
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
import os
import sys
from tqdm import tqdm
import wandb
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.stats import pearsonr, spearmanr
import json
from transformers import EsmTokenizer, EsmModel

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from protein_drug_discovery.models.doublesg_integration import (
    DoubleSGDTAModel, MolecularGraphProcessor
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
cl
ass EnhancedDoubleSGModel(nn.Module):
    """Enhanced DoubleSG-DTA model with ESM-2 integration"""
    
    def __init__(self, 
                 esm_model_name: str = "facebook/esm2_t12_35M_UR50D",
                 drug_feature_dim: int = 78,
                 gin_hidden_dim: int = 128,
                 gin_layers: int = 5,
                 protein_cnn_filters: int = 32,
                 embed_dim: int = 128,
                 output_dim: int = 128,
                 dropout: float = 0.2,
                 use_esm: bool = True):
        
        super(EnhancedDoubleSGModel, self).__init__()
        
        self.use_esm = use_esm
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        
        # ESM-2 protein encoder (optional)
        if self.use_esm:
            self.esm_tokenizer = EsmTokenizer.from_pretrained(esm_model_name)
            self.esm_model = EsmModel.from_pretrained(esm_model_name)
            self.esm_hidden_dim = self.esm_model.config.hidden_size
            
            # Freeze ESM-2 parameters for efficiency
            for param in self.esm_model.parameters():
                param.requires_grad = False
        
        # Drug graph encoder (GIN layers from DoubleSG-DTA)
        from torch_geometric.nn import GINConv, global_add_pool, global_max_pool
        from torch.nn import Sequential, Linear, ReLU
        
        dim = gin_hidden_dim
        
        # GIN convolution layers
        nn1 = Sequential(Linear(drug_feature_dim, dim), ReLU(), Linear(dim, dim))
        self.conv1 = GINConv(nn1)
        self.bn1 = torch.nn.BatchNorm1d(dim)

        nn2 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv2 = GINConv(nn2)
        self.bn2 = torch.nn.BatchNorm1d(dim)

        nn3 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv3 = GINConv(nn3)
        self.bn3 = torch.nn.BatchNorm1d(dim)

        nn4 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv4 = GINConv(nn4)
        self.bn4 = torch.nn.BatchNorm1d(dim)

        nn5 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv5 = GINConv(nn5)
        self.bn5 = torch.nn.BatchNorm1d(dim)
        
        self.fc1_xd = Linear(dim, output_dim)
        
        # Protein sequence processing
        if not self.use_esm:
            self.embedding_xt = nn.Embedding(26, embed_dim)  # 25 amino acids + padding
            self.conv_xt1 = nn.Conv1d(in_channels=1000, out_channels=protein_cnn_filters, kernel_size=8)
            self.SE_protein = self._create_se_block(protein_cnn_filters)
            self.fc_xt1 = nn.Linear(protein_cnn_filters * 121, output_dim)
        else:
            # ESM-2 protein processing
            self.protein_projection = nn.Linear(self.esm_hidden_dim, output_dim)
        
        # Drug SMILES sequence processing (from DoubleSG-DTA)
        self.embedding_xt_smile = nn.Embedding(100, embed_dim)
        self.conv_xt2 = nn.Conv1d(in_channels=100, out_channels=protein_cnn_filters, kernel_size=8)
        self.SE_drug = self._create_se_block(protein_cnn_filters)
        self.fc_xt2 = nn.Linear(protein_cnn_filters * 121, output_dim)
        
        # Cross-attention mechanism
        self.cross_attention = self._create_cross_attention(output_dim, output_dim)
        
        # Final prediction layers
        total_features = output_dim * 3  # drug_graph + protein + drug_smiles
        self.fc1 = nn.Linear(total_features, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.out = nn.Linear(256, 1)  # Regression output
    
    def _create_se_block(self, channels: int) -> nn.Module:
        """Create Squeeze-and-Excitation block"""
        return nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, channels // 16, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(channels // 16, channels, kernel_size=1),
            nn.Sigmoid()
        )
    
    def _create_cross_attention(self, drug_dim: int, protein_dim: int, num_heads: int = 8) -> nn.Module:
        """Create cross-attention mechanism"""
        return nn.MultiheadAttention(
            embed_dim=drug_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )