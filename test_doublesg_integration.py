#!/usr/bin/env python3
"""
Test script for DoubleSG-DTA integration
Quick test to verify all components are working
"""

import torch
import sys
import os

# Add current directory to Python path
sys.path.append(os.getcwd())

def test_integration():
    print("🧪 Testing DoubleSG-DTA Integration")
    print("=" * 40)
    
    try:
        # Test imports
        print("1. Testing imports...")
        from protein_drug_discovery.models.doublesg_integration import (
            DoubleSGDTAModel, MolecularGraphProcessor
        )
        print("   ✅ Imports successful")
        
        # Test molecular graph processing
        print("2. Testing molecular graph processing...")
        processor = MolecularGraphProcessor()
        
        # Test with simple molecule (methane)
        smiles = "C"
        graph = processor.smiles_to_graph(smiles)
        
        print(f"   ✅ Graph created: {graph['num_nodes']} nodes, {graph['edge_index'].shape[1]} edges")
        
        # Test with drug molecule (ibuprofen)
        smiles = "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O"
        graph = processor.smiles_to_graph(smiles)
        
        print(f"   ✅ Drug graph created: {graph['num_nodes']} nodes, {graph['edge_index'].shape[1]} edges")
        
        # Test batch processing
        print("3. Testing batch processing...")
        smiles_list = ["C", "CC", "CCC"]
        batch_graph = processor.batch_smiles_to_graphs(smiles_list)
        
        print(f"   ✅ Batch graph created: {batch_graph['x'].shape[0]} total nodes")
        
        # Test model initialization
        print("4. Testing model initialization...")
        
        # Create mock ESM model
        class MockESMModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                
            def forward(self, input_ids, attention_mask=None, **kwargs):
                batch_size, seq_len = input_ids.shape
                last_hidden_state = torch.randn(batch_size, seq_len, 480)
                return type('MockOutput', (), {'last_hidden_state': last_hidden_state})()
        
        esm_model = MockESMModel()
        
        model = DoubleSGDTAModel(
            esm_model=esm_model,
            drug_feature_dim=78,
            esm_hidden_dim=480,
            gin_hidden_dim=64,  # Smaller for test
            gin_layers=2,       # Fewer layers for test
            attention_heads=2,  # Fewer heads for test
            final_hidden_dim=64
        )
        
        print(f"   ✅ Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
        
        # Test forward pass
        print("5. Testing forward pass...")
        
        # Create mock inputs
        protein_tokens = {
            'input_ids': torch.randint(0, 1000, (1, 100)),
            'attention_mask': torch.ones(1, 100)
        }
        
        drug_graph_data = (
            graph['node_features'],
            graph['edge_index'],
            torch.zeros(graph['num_nodes'], dtype=torch.long)
        )
        
        model.eval()
        with torch.no_grad():
            predictions = model(protein_tokens, drug_graph_data)
            
        print(f"   ✅ Forward pass successful")
        print(f"   📊 Affinity prediction: {predictions['affinity'].item():.3f}")
        print(f"   📊 Toxicity prediction: {predictions['toxicity'].item():.3f}")
        print(f"   📊 Solubility prediction: {predictions['solubility'].item():.3f}")
        
        print("\n🎉 All tests passed!")
        print("✅ DoubleSG-DTA integration is working correctly")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = test_integration()
    exit(0 if success else 1)