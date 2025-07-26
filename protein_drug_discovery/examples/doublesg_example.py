"""
DoubleSG-DTA Integration Example
Complete example showing how to use the enhanced DoubleSG-DTA model with ESM-2
"""

import torch
import torch.nn as nn
from transformers import EsmTokenizer, EsmModel
import pandas as pd
import numpy as np
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from protein_drug_discovery.models.doublesg_integration import (
    DoubleSGDTAModel, MolecularGraphProcessor
)
from protein_drug_discovery.training.doublesg_trainer import (
    DoubleSGDTATrainer, DrugTargetAffinityDataset, create_doublesg_datasets
)

def main():
    """Main example function"""
    
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Step 1: Initialize ESM-2 model
    print("Loading ESM-2 model...")
    tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
    esm_model = EsmModel.from_pretrained("facebook/esm2_t6_8M_UR50D")
    
    # Step 2: Initialize enhanced DoubleSG-DTA model
    print("Initializing DoubleSG-DTA model...")
    model = DoubleSGDTAModel(
        esm_model=esm_model,
        drug_feature_dim=78,
        esm_hidden_dim=320,  # ESM-2 8M hidden dimension
        gin_hidden_dim=128,
        gin_layers=3,
        attention_heads=8,
        final_hidden_dim=256
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Step 3: Example inference
    print("Running example inference...")
    
    # Sample data
    sample_smiles = "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O"  # Ibuprofen
    sample_protein = "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"
    
    # Process inputs
    graph_processor = MolecularGraphProcessor()
    drug_graph = graph_processor.smiles_to_graph(sample_smiles)
    
    protein_tokens = tokenizer(
        sample_protein,
        max_length=512,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    # Create batch format
    drug_graph_batch = (
        drug_graph['node_features'].unsqueeze(0),
        drug_graph['edge_index'],
        torch.zeros(drug_graph['num_nodes'], dtype=torch.long)
    )
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        predictions = model(protein_tokens, drug_graph_batch)
    
    print(f"Predicted affinity: {predictions['affinity'].item():.4f}")
    print(f"Predicted toxicity: {predictions['toxicity'].item():.4f}")
    print(f"Predicted solubility: {predictions['solubility'].item():.4f}")

if __name__ == "__main__":
    main()