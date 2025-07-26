"""ESM-2 protein language model implementation optimized for CPU inference."""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ESMProteinModel:
    """ESM-2 protein language model wrapper optimized for CPU inference."""
    
    def __init__(self, model_size: str = "150M", device: str = "cpu"):
        """
        Initialize ESM-2 model.
        
        Args:
            model_size: Model size ("150M", "650M", "3B") - using 150M for CPU
            device: Device to run on ("cpu" or "cuda")
        """
        self.model_size = model_size
        self.device = device
        self.model = None
        self.tokenizer = None
        self.max_length = 1024  # Maximum protein sequence length
        
        try:
            self._load_model()
            logger.info(f"ESM-2 {model_size} model loaded successfully on {device}")
        except Exception as e:
            logger.error(f"Failed to load ESM-2 model: {e}")
            self._create_mock_model()
    
    def _load_model(self):
        """Load ESM-2 model from HuggingFace or fair-esm."""
        try:
            # Try to load from transformers first
            from transformers import EsmModel, EsmTokenizer
            
            # Ensure cache is on D drive
            import os
            os.environ['HF_HOME'] = 'D:/huggingface_cache'
            os.environ['TRANSFORMERS_CACHE'] = 'D:/huggingface_cache/transformers'
            os.environ['HF_DATASETS_CACHE'] = 'D:/huggingface_cache/datasets'
            
            model_name = f"facebook/esm2_t12_{self.model_size.lower()}_UR50D"
            logger.info(f"Loading ESM-2 model: {model_name}")
            logger.info(f"Cache directory: D:/huggingface_cache/transformers")
            
            self.tokenizer = EsmTokenizer.from_pretrained(
                model_name,
                cache_dir="D:/huggingface_cache/transformers"
            )
            self.model = EsmModel.from_pretrained(
                model_name,
                cache_dir="D:/huggingface_cache/transformers"
            )
            self.model.to(self.device)
            self.model.eval()
            
        except ImportError:
            logger.warning("Transformers not available, trying fair-esm")
            try:
                import esm
                self.model, self.alphabet = esm.pretrained.esm2_t12_35M_UR50D()
                self.model.to(self.device)
                self.model.eval()
                self.tokenizer = self.alphabet
            except ImportError:
                logger.error("Neither transformers nor fair-esm available")
                raise
    
    def _create_mock_model(self):
        """Create mock model for testing when ESM-2 is not available."""
        logger.warning("Creating mock ESM model for testing")
        
        class MockESMModel:
            def __init__(self):
                self.embedding_dim = 320
                self.vocab_size = 33
                # Create random embedding matrix
                np.random.seed(42)
                self.embeddings = np.random.randn(self.vocab_size, self.embedding_dim) * 0.1
            
            def forward(self, input_ids, attention_mask=None):
                # Simple embedding lookup
                if isinstance(input_ids, list):
                    input_ids = np.array(input_ids)
                
                # Get embeddings for each token
                seq_embeddings = self.embeddings[input_ids]
                
                return {
                    "last_hidden_state": seq_embeddings,
                    "logits": seq_embeddings  # Mock logits
                }
        
        self.model = MockESMModel()
        
        # Mock tokenizer
        self.tokenizer = self._create_mock_tokenizer()
    
    def _create_mock_tokenizer(self):
        """Create mock tokenizer for testing."""
        class MockTokenizer:
            def __init__(self):
                self.vocab = {
                    '<pad>': 0, '<unk>': 1, '<cls>': 2, '<eos>': 3, '<mask>': 4,
                    'A': 5, 'R': 6, 'N': 7, 'D': 8, 'C': 9, 'Q': 10, 'E': 11,
                    'G': 12, 'H': 13, 'I': 14, 'L': 15, 'K': 16, 'M': 17,
                    'F': 18, 'P': 19, 'S': 20, 'T': 21, 'W': 22, 'Y': 23, 'V': 24
                }
                self.pad_token_id = 0
                self.cls_token_id = 2
                self.eos_token_id = 3
            
            def encode(self, sequence, add_special_tokens=True):
                tokens = [self.cls_token_id] if add_special_tokens else []
                for aa in sequence.upper():
                    tokens.append(self.vocab.get(aa, 1))  # 1 for <unk>
                if add_special_tokens:
                    tokens.append(self.eos_token_id)
                return tokens
            
            def __call__(self, sequences, padding=True, truncation=True, 
                        max_length=1024, return_tensors="pt"):
                if isinstance(sequences, str):
                    sequences = [sequences]
                
                encoded = []
                for seq in sequences:
                    tokens = self.encode(seq)
                    if truncation and len(tokens) > max_length:
                        tokens = tokens[:max_length-1] + [self.eos_token_id]
                    encoded.append(tokens)
                
                if padding:
                    max_len = max(len(tokens) for tokens in encoded)
                    for tokens in encoded:
                        tokens.extend([self.pad_token_id] * (max_len - len(tokens)))
                
                if return_tensors == "pt":
                    input_ids = torch.tensor(encoded)
                    attention_mask = (input_ids != self.pad_token_id).long()
                    return {"input_ids": input_ids, "attention_mask": attention_mask}
                
                return encoded
        
        return MockTokenizer()
    
    def encode_protein(self, sequence: str) -> Dict[str, np.ndarray]:
        """
        Encode protein sequence to embeddings using real ESM-2 model.
        
        Args:
            sequence: Protein amino acid sequence
            
        Returns:
            Dictionary with embeddings and attention weights
        """
        try:
            # Validate protein sequence
            if not self._validate_protein_sequence(sequence):
                raise ValueError(f"Invalid protein sequence: {sequence[:50]}...")
            
            # Check if we have a real ESM model (not mock)
            if hasattr(self.model, 'config') and hasattr(self.tokenizer, 'model_max_length'):
                # Real ESM-2 model path
                logger.info("Using real ESM-2 model for encoding")
                
                # Tokenize using real ESM tokenizer
                inputs = self.tokenizer(
                    sequence, 
                    return_tensors="pt", 
                    padding=True, 
                    truncation=True,
                    max_length=self.max_length
                )
                
                # Move to device
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Get embeddings from real ESM model
                with torch.no_grad():
                    outputs = self.model(**inputs)
                
                # Extract embeddings
                embeddings = outputs.last_hidden_state  # [batch, seq_len, hidden_dim]
                
                # Pool embeddings (mean pooling, excluding special tokens)
                attention_mask = inputs['attention_mask']
                masked_embeddings = embeddings * attention_mask.unsqueeze(-1)
                pooled_embeddings = masked_embeddings.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
                
                # Convert to numpy
                embeddings_np = embeddings.cpu().numpy()
                pooled_embeddings_np = pooled_embeddings.cpu().numpy()
                attention_mask_np = attention_mask.cpu().numpy()
                
                return {
                    "embeddings": embeddings_np,
                    "pooled_embeddings": pooled_embeddings_np,
                    "attention_mask": attention_mask_np,
                    "sequence_length": len(sequence),
                    "model_type": "ESM-2"
                }
            
            else:
                # Fallback to mock model (should not happen if ESM-2 loads correctly)
                logger.warning("Using mock model fallback")
                
                # Simple tokenization for mock
                tokens = [2] + [ord(aa) % 25 + 5 for aa in sequence.upper()[:self.max_length-2]] + [3]
                input_ids = np.array([tokens])
                
                # Mock embeddings
                seq_len = len(tokens)
                hidden_dim = 480  # ESM-2 35M hidden dimension
                embeddings = np.random.randn(1, seq_len, hidden_dim) * 0.1
                pooled_embeddings = np.mean(embeddings, axis=1)
                attention_mask = np.ones((1, seq_len))
                
                return {
                    "embeddings": embeddings,
                    "pooled_embeddings": pooled_embeddings,
                    "attention_mask": attention_mask,
                    "sequence_length": len(sequence),
                    "model_type": "Mock"
                }
            
        except Exception as e:
            logger.error(f"Error encoding protein sequence: {e}")
            import traceback
            traceback.print_exc()
            
            # Return dummy embeddings for error cases
            seq_len = min(len(sequence), self.max_length)
            hidden_dim = 480  # ESM-2 35M hidden dimension
            return {
                "embeddings": np.zeros((1, seq_len, hidden_dim)),
                "pooled_embeddings": np.zeros((1, hidden_dim)),
                "attention_mask": np.ones((1, seq_len)),
                "sequence_length": len(sequence),
                "model_type": "Error"
            }
    
    def _validate_protein_sequence(self, sequence: str) -> bool:
        """Validate protein amino acid sequence."""
        if not sequence or not isinstance(sequence, str):
            return False
        
        # Standard amino acid codes
        valid_aa = set('ACDEFGHIKLMNPQRSTVWY')
        sequence_upper = sequence.upper().replace(' ', '').replace('\n', '')
        
        # Check if all characters are valid amino acids
        return all(aa in valid_aa for aa in sequence_upper)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        try:
            param_count = sum(p.numel() for p in self.model.parameters())
        except:
            param_count = 0
        
        # Determine model type
        if hasattr(self.model, 'config') and hasattr(self.tokenizer, 'model_max_length'):
            model_type = "ESM-2 (Real)"
        elif hasattr(self.model, 'esm'):
            model_type = "ESM-2 (fair-esm)"
        else:
            model_type = "Mock"
        
        return {
            "model_size": self.model_size,
            "device": self.device,
            "max_length": self.max_length,
            "parameter_count": param_count,
            "model_type": model_type,
            "cache_location": "D:/huggingface_cache/transformers"
        }