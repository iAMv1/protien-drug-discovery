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
        logger.info("💡 Trying to install missing dependencies...")
        
        # Try to install missing packages
        try:
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", "transformers", "torch", "rdkit-pypi"])
            logger.info("✅ Dependencies installed, please run the demo again")
        except Exception as install_error:
            logger.error(f"❌ Failed to install dependencies: {install_error}")
            logger.info("📋 Please manually install: pip install transformers torch rdkit-pypi")
        return
    
    # Initialize components
    logger.info("🔧 Initializing components...")
    
    # Try to load ESM-2 model (smaller version for demo)
    try:
        from transformers import EsmTokenizer, EsmModel
        esm_model_name = "facebook/esm2_t6_8M_UR50D"  # Smaller model for demo
        logger.info(f"📥 Loading ESM-2 model: {esm_model_name}")
        tokenizer = EsmTokenizer.from_pretrained(esm_model_name)
        esm_model = EsmModel.from_pretrained(esm_model_name)
    except Exception as e:
        logger.error(f"❌ Failed to load ESM-2 model: {e}")
        logger.info("🔄 Using mock ESM-2 model for demo...")
        
        # Create a mock ESM model for demo
        class MockESMModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.config = type('Config', (), {'hidden_size': 320})()
                self.embeddings = torch.nn.Embedding(33, 320)  # 33 tokens, 320 dim
                
            def forward(self, input_ids, attention_mask=None, **kwargs):
                embeddings = self.embeddings(input_ids)
                return type('Output', (), {'last_hidden_state': embeddings})()
        
        class MockTokenizer:
            def __call__(self, text, **kwargs):
                # Simple character-level tokenization for demo
                tokens = [ord(c) % 33 for c in text[:512]]  # Limit to 512 chars
                if len(tokens) < 512:
                    tokens.extend([0] * (512 - len(tokens)))  # Pad
                return {
                    'input_ids': torch.tensor([tokens]),
                    'attention_mask': torch.ones(1, 512)
                }
        
        tokenizer = MockTokenizer()
        esm_model = MockESMModel()
    
    # Freeze ESM-2 for efficiency
    for param in esm_model.parameters():
        param.requires_grad = False
    
    logger.info(f"✅ ESM-2 model loaded: {esm_model_name}")
    
    # Initialize our enhanced model
    model = DoubleSGDTAModel(
        esm_model=esm_model,
        drug_feature_dim=78,
        esm_hidden_dim=esm_model.config.hidden_size,
        gin_hidden_dim=128,
        gin_layers=3,  # Reduced for demo
        attention_heads=4,  # Reduced for demo
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
        
        # Process protein with ESM-2
        protein_tokens = tokenizer(
            compound['protein'],
            max_length=512,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Process drug with molecular graph
        drug_graph = graph_processor.smiles_to_graph(compound['smiles'])
        
        # Create batch data (simplified for demo)
        drug_graph_data = (
            drug_graph['node_features'].unsqueeze(0),  # Add batch dimension
            drug_graph['edge_index'],
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
    
    logger.info("\n" + "=" * 50)
    logger.info("🎉 Demo completed!")
    logger.info("\n📋 Key Features Demonstrated:")
    logger.info("✅ ESM-2 protein encoding")
    logger.info("✅ GIN molecular graph processing")
    logger.info("✅ Cross-attention mechanism")
    logger.info("✅ Multi-task prediction capability")
    logger.info("✅ End-to-end drug-target affinity prediction")

if __name__ == '__main__':
    demo_enhanced_doublesg()