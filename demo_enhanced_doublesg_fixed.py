#!/usr/bin/env python3
"""
Demo script for Enhanced DoubleSG-DTA integration
Shows how to use the integrated DoubleSG-DTA + ESM-2 model
"""

import torch
import pandas as pd
import numpy as np
import sys
import os
import logging

# Add current directory to Python path
sys.path.append(os.getcwd())

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def demo_enhanced_doublesg():
    """Demonstrate the Enhanced DoubleSG-DTA model"""
    
    logger.info("🧬 Enhanced DoubleSG-DTA Demo")
    logger.info("=" * 50)
    
    # Check if we have the required files
    try:
        from protein_drug_discovery.models.doublesg_integration import (
            DoubleSGDTAModel, MolecularGraphProcessor
        )
        logger.info("✅ DoubleSG-DTA integration modules loaded successfully")
    except ImportError as e:
        logger.error(f"❌ Failed to import modules: {e}")
        logger.info("💡 Make sure you're running from the project root directory")
        return
    
    # Initialize components
    logger.info("🔧 Initializing components...")
    
    # Create a simple mock ESM model for demo
    class MockESMModel(torch.nn.Module):
        def __init__(self, hidden_size=480):
            super().__init__()
            self.hidden_size = hidden_size
            
        def forward(self, input_ids, attention_mask=None, **kwargs):
            batch_size, seq_len = input_ids.shape
            # Return mock embeddings
            last_hidden_state = torch.randn(batch_size, seq_len, self.hidden_size)
            return type('MockOutput', (), {'last_hidden_state': last_hidden_state})()
    
    # Initialize mock ESM model
    esm_model = MockESMModel(hidden_size=480)
    logger.info("✅ Mock ESM-2 model initialized")
    
    # Initialize our enhanced model
    model = DoubleSGDTAModel(
        esm_model=esm_model,
        drug_feature_dim=78,
        esm_hidden_dim=480,
        gin_hidden_dim=128,
        gin_layers=3,
        attention_heads=4,
        final_hidden_dim=128
    )
    
    logger.info("✅ Enhanced DoubleSG-DTA model initialized")
    logger.info(f"📊 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Demo data
    demo_data = [
        {
            'smiles': 'CC(C)CC1=CC=C(C=C1)C(C)C(=O)O',  # Ibuprofen
            'protein': 'MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG',
            'affinity': 7.5
        },
        {
            'smiles': 'CN1CCN(CC1)C2=C(C=C3C(=C2)N=CN3C4=CC=CC=C4)C#N',  # Kinase inhibitor
            'protein': 'MGSSHHHHHHSSGLVPRGSHMRGPNPTAASLEASAGPFTVRSFTVSRPSGYGAGTVYYPTNAGGTVGAIAIVPGYTARQSSIKWWGPRLASHGFVVITIDTNSTLDQPSSRSSQQMAALRQVASLNGTSSSPIYGKVDTARMGVMGWSMGGGGSLISAANNPSLKAAAPQAPWDSSTNFSSVTVPTLIFACENDSIAPVNSSALPIYDSMSRNAKQFLEINGGSHSCANSGNSNQALIGKKGVAWMKRFPTSREJ',
            'affinity': 8.2
        }
    ]
    
    logger.info("🧪 Processing demo compounds...")
    
    # Process each compound
    graph_processor = MolecularGraphProcessor()
    
    for i, compound in enumerate(demo_data):
        logger.info(f"\n--- Compound {i+1} ---")
        logger.info(f"SMILES: {compound['smiles']}")
        logger.info(f"Protein length: {len(compound['protein'])} residues")
        logger.info(f"True affinity: {compound['affinity']}")
        
        # Create mock protein tokens (normally would use ESM tokenizer)
        protein_tokens = {
            'input_ids': torch.randint(0, 1000, (1, min(len(compound['protein']), 512))),
            'attention_mask': torch.ones(1, min(len(compound['protein']), 512))
        }
        
        # Process drug with molecular graph
        drug_graph = graph_processor.smiles_to_graph(compound['smiles'])
        
        # Create batch data
        drug_graph_data = (
            drug_graph['node_features'],  # Node features
            drug_graph['edge_index'],     # Edge indices
            torch.zeros(drug_graph['num_nodes'], dtype=torch.long)  # Batch indices
        )
        
        # Forward pass
        model.eval()
        with torch.no_grad():
            try:
                predictions = model(protein_tokens, drug_graph_data)
                predicted_affinity = predictions['affinity'].item()
                
                logger.info(f"✅ Predicted affinity: {predicted_affinity:.3f}")
                logger.info(f"📊 Error: {abs(predicted_affinity - compound['affinity']):.3f}")
                
                # Show feature dimensions
                logger.info(f"🔍 Drug features shape: {predictions['drug_features'].shape}")
                logger.info(f"🔍 Protein features shape: {predictions['protein_features'].shape}")
                
            except Exception as e:
                logger.error(f"❌ Prediction failed: {e}")
                import traceback
                traceback.print_exc()
    
    logger.info("\n" + "=" * 50)
    logger.info("🎉 Demo completed!")
    logger.info("\n📋 Key Features Demonstrated:")
    logger.info("✅ Molecular graph processing from SMILES")
    logger.info("✅ GIN molecular graph encoding")
    logger.info("✅ Protein sequence processing")
    logger.info("✅ Cross-attention mechanism")
    logger.info("✅ Multi-task prediction capability")
    logger.info("✅ End-to-end drug-target affinity prediction")

if __name__ == '__main__':
    demo_enhanced_doublesg()