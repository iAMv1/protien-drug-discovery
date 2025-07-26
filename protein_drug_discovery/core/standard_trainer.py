# protein_drug_discovery/core/standard_trainer.py

import torch
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from pathlib import Path
import json
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# Standard transformers and TRL imports
try:
    from transformers import (
        AutoTokenizer, 
        AutoModelForCausalLM, 
        TrainingArguments,
        BitsAndBytesConfig
    )
    from trl import SFTTrainer
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, TaskType
    TRANSFORMERS_AVAILABLE = True
except ImportError as e:
    TRANSFORMERS_AVAILABLE = False
    logging.error(f"Required libraries not available: {e}")

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    logging.warning("Weights & Biases not available - install with: pip install wandb")

logging.basicConfig(level=logging.INFO)

class ProteinDrugDatasetFormatter:
    """Format protein-drug interaction data for training"""
    
    def __init__(self):
        self.instruction_template = """You are a protein-drug interaction prediction expert. Given a protein sequence and drug SMILES, predict the binding affinity and properties.

### Protein Sequence:
{protein_sequence}

### Drug SMILES:
{drug_smiles}

### Task:
Predict the binding affinity (pKd), binding probability, and drug properties."""

        self.response_template = """Based on the protein-drug interaction analysis:

**Binding Affinity (pKd):** {pkd:.2f}
**Binding Probability:** {binding_prob:.3f}
**Confidence Score:** {confidence:.3f}
**Predicted Properties:**
- Toxicity Risk: {toxicity_risk}
- Solubility: {solubility_level}
- Drug-likeness: {druglikeness}

**Explanation:** This prediction is based on protein-drug interaction patterns, molecular properties, and binding site analysis."""

    def format_training_data(self, df: pd.DataFrame) -> List[Dict[str, str]]:
        """
        Format protein-drug interaction data for instruction tuning.
        
        Args:
            df: DataFrame with columns: protein, compound, pKd, binding_label
            
        Returns:
            List of formatted training examples
        """
        formatted_data = []
        
        for _, row in df.iterrows():
            try:
                # Create instruction
                instruction = self.instruction_template.format(
                    protein_sequence=row['protein'][:500],  # Truncate long sequences
                    drug_smiles=row['compound']
                )
                
                # Create response with predictions
                binding_prob = 1.0 if row['binding_label'] == 1 else 0.3
                confidence = min(0.9, max(0.1, 1.0 - abs(row['pKd'] - 6.0) / 6.0))
                
                # Categorize properties
                toxicity_risk = "Low" if row['pKd'] > 6.5 else "Medium" if row['pKd'] > 5.0 else "High"
                solubility_level = "Good" if row['pKd'] < 8.0 else "Moderate"
                druglikeness = "High" if 5.0 < row['pKd'] < 9.0 else "Moderate"
                
                response = self.response_template.format(
                    pkd=row['pKd'],
                    binding_prob=binding_prob,
                    confidence=confidence,
                    toxicity_risk=toxicity_risk,
                    solubility_level=solubility_level,
                    druglikeness=druglikeness
                )
                
                # Format for training (simple text format)
                formatted_text = f"{instruction}\n\n{response}"
                formatted_data.append({"text": formatted_text})
                
            except Exception as e:
                logging.warning(f"Error formatting row: {e}")
                continue
        
        logging.info(f"Formatted {len(formatted_data)} training examples")
        return formatted_data

class StandardProteinTrainer:
    """Standard transformers-based QLoRA trainer for protein-drug interaction prediction"""
    
    def __init__(self, 
                 model_name: str = "microsoft/DialoGPT-small",
                 max_seq_length: int = 1024,
                 load_in_4bit: bool = True):
        """
        Initialize standard QLoRA trainer.
        
        Args:
            model_name: HuggingFace model name
            max_seq_length: Maximum sequence length
            load_in_4bit: Use 4-bit quantization (QLoRA)
        """
        self.model_name = model_name
        self.max_seq_length = max_seq_length
        self.load_in_4bit = load_in_4bit
        self.model = None
        self.tokenizer = None
        self.formatter = ProteinDrugDatasetFormatter()
        
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Transformers and TRL are required. Install with: pip install transformers trl peft")
        
        self._load_model()
    
    def _load_model(self):
        """Load model with QLoRA configuration"""
        try:
            logging.info(f"Loading model: {self.model_name}")
            
            # Configure quantization
            if self.load_in_4bit:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
            else:
                quantization_config = None
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                padding_side="right"
            )
            
            # Add pad token if missing
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=quantization_config,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            )
            
            # Configure LoRA
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=16,  # LoRA rank
                lora_alpha=32,  # LoRA alpha
                lora_dropout=0.1,
                target_modules=["c_attn", "c_proj"],  # For DialoGPT
                bias="none",
            )
            
            # Apply LoRA
            self.model = get_peft_model(self.model, lora_config)
            
            logging.info("Standard QLoRA model loaded successfully")
            
        except Exception as e:
            logging.error(f"Failed to load model: {e}")
            raise
    
    def prepare_dataset(self, train_df: pd.DataFrame, val_df: Optional[pd.DataFrame] = None) -> Tuple[Dataset, Optional[Dataset]]:
        """
        Prepare datasets for training.
        
        Args:
            train_df: Training DataFrame
            val_df: Validation DataFrame (optional)
            
        Returns:
            Training and validation datasets
        """
        # Format training data
        train_formatted = self.formatter.format_training_data(train_df)
        train_dataset = Dataset.from_list(train_formatted)
        
        val_dataset = None
        if val_df is not None:
            val_formatted = self.formatter.format_training_data(val_df)
            val_dataset = Dataset.from_list(val_formatted)
        
        logging.info(f"Prepared datasets - Train: {len(train_dataset)}, Val: {len(val_dataset) if val_dataset else 0}")
        return train_dataset, val_dataset
    
    def train(self, 
              train_dataset: Dataset,
              val_dataset: Optional[Dataset] = None,
              output_dir: str = "./models/protein_drug_model",
              num_train_epochs: int = 1,
              per_device_train_batch_size: int = 2,
              gradient_accumulation_steps: int = 4,
              warmup_steps: int = 5,
              learning_rate: float = 2e-4,
              logging_steps: int = 1,
              optim: str = "adamw_torch",
              weight_decay: float = 0.01,
              lr_scheduler_type: str = "linear",
              seed: int = 3407,
              use_wandb: bool = False):
        """
        Train the model using standard QLoRA.
        """
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize wandb if requested
        if use_wandb and WANDB_AVAILABLE:
            wandb.init(
                project="protein-drug-discovery",
                name=f"standard-qlora-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
                config={
                    "model_name": self.model_name,
                    "max_seq_length": self.max_seq_length,
                    "num_train_epochs": num_train_epochs,
                    "batch_size": per_device_train_batch_size,
                    "learning_rate": learning_rate,
                }
            )
        
        # Training arguments
        training_args = TrainingArguments(
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=warmup_steps,
            num_train_epochs=num_train_epochs,
            learning_rate=learning_rate,
            fp16=torch.cuda.is_available(),
            logging_steps=logging_steps,
            optim=optim,
            weight_decay=weight_decay,
            lr_scheduler_type=lr_scheduler_type,
            seed=seed,
            output_dir=str(output_path),
            report_to="wandb" if (use_wandb and WANDB_AVAILABLE) else None,
            save_strategy="epoch",
            evaluation_strategy="epoch" if val_dataset else "no",
            load_best_model_at_end=True if val_dataset else False,
            save_total_limit=2,
            remove_unused_columns=False,
        )
        
        # Create trainer
        trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            dataset_text_field="text",
            max_seq_length=self.max_seq_length,
            args=training_args,
        )
        
        # Start training
        logging.info("Starting standard QLoRA training...")
        trainer_stats = trainer.train()
        
        # Save model
        self.save_model(str(output_path))
        
        if use_wandb and WANDB_AVAILABLE:
            wandb.finish()
        
        return trainer_stats
    
    def save_model(self, output_dir: str):
        """Save the trained QLoRA model"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save LoRA adapter
        self.model.save_pretrained(output_path / "lora_adapter")
        self.tokenizer.save_pretrained(output_path / "tokenizer")
        
        # Save configuration
        config = {
            "model_name": self.model_name,
            "max_seq_length": self.max_seq_length,
            "load_in_4bit": self.load_in_4bit,
            "training_completed": datetime.now().isoformat(),
            "model_type": "Standard QLoRA"
        }
        
        with open(output_path / "training_config.json", 'w') as f:
            json.dump(config, f, indent=2)
        
        logging.info(f"QLoRA model saved to {output_path}")
    
    def inference(self, protein_sequence: str, drug_smiles: str, max_new_tokens: int = 256) -> str:
        """
        Run inference on protein-drug pair.
        """
        # Format input
        instruction = self.formatter.instruction_template.format(
            protein_sequence=protein_sequence[:500],
            drug_smiles=drug_smiles
        )
        
        # Tokenize input
        inputs = self.tokenizer(
            instruction,
            return_tensors="pt",
            max_length=self.max_seq_length,
            truncation=True,
            padding=True
        )
        
        if torch.cuda.is_available():
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        # Generate response
        self.model.eval()
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode response
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the generated part
        generated_text = response[len(instruction):].strip()
        
        return generated_text
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        if self.model is None:
            return {"error": "Model not loaded"}
        
        try:
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in self.model.parameters())
        except:
            trainable_params = 0
            total_params = 0
        
        return {
            "model_name": self.model_name,
            "model_type": "Standard QLoRA",
            "max_seq_length": self.max_seq_length,
            "load_in_4bit": self.load_in_4bit,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "trainable_percentage": (100 * trainable_params / total_params) if total_params > 0 else 0,
            "libraries_available": {
                "transformers": TRANSFORMERS_AVAILABLE,
                "wandb": WANDB_AVAILABLE
            }
        }

def main():
    """Test the standard protein trainer"""
    print("=" * 60)
    print("🧬 STANDARD PROTEIN TRAINER TEST")
    print("=" * 60)
    
    if not TRANSFORMERS_AVAILABLE:
        print("❌ Required libraries not available. Install with:")
        print("   pip install transformers trl peft bitsandbytes")
        return
    
    try:
        # Test dataset formatter
        print("📝 Testing dataset formatter...")
        formatter = ProteinDrugDatasetFormatter()
        
        # Create sample data
        sample_data = pd.DataFrame({
            'protein': [
                'MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG',
                'MAVPFVEDWDLVQTLGEGAYGEVQLAVNRVTEEAVAVKIVDMKRAVDCPE'
            ],
            'compound': [
                'COC1=C(C=C2C(=C1)N=CN=C2NC3=CC(=C(C=C3)F)Cl)OCCCN4CCOCC4',
                'CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)NC4=NC=CC(=N4)C5=CN=CC=C5'
            ],
            'pKd': [6.5, 7.2],
            'binding_label': [1, 1]
        })
        
        formatted_data = formatter.format_training_data(sample_data)
        print(f"  ✅ Formatted {len(formatted_data)} training examples")
        
        # Show sample
        if formatted_data:
            sample = formatted_data[0]
            print(f"\n📋 Sample formatted text:")
            print(f"  {sample['text'][:200]}...")
        
        # Initialize trainer
        print(f"\n🚀 Testing trainer initialization...")
        trainer = StandardProteinTrainer(
            model_name="microsoft/DialoGPT-small",
            max_seq_length=1024,
            load_in_4bit=False  # Disable for testing
        )
        
        # Get model info
        info = trainer.get_model_info()
        print(f"\n📊 Model Information:")
        print(f"  Model: {info['model_name']}")
        print(f"  Type: {info['model_type']}")
        print(f"  Max Length: {info['max_seq_length']}")
        print(f"  4-bit: {info['load_in_4bit']}")
        print(f"  Total Parameters: {info['total_parameters']:,}")
        print(f"  Trainable Parameters: {info['trainable_parameters']:,}")
        print(f"  Trainable %: {info['trainable_percentage']:.2f}%")
        
        # Test dataset preparation
        print(f"\n📝 Testing dataset preparation...")
        train_dataset, val_dataset = trainer.prepare_dataset(sample_data)
        print(f"  ✅ Dataset prepared: {len(train_dataset)} samples")
        
        # Test inference
        print(f"\n🔮 Testing inference...")
        test_protein = "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"
        test_drug = "COC1=C(C=C2C(=C1)N=CN=C2NC3=CC(=C(C=C3)F)Cl)OCCCN4CCOCC4"
        
        prediction = trainer.inference(test_protein, test_drug, max_new_tokens=64)
        print(f"  Prediction: {prediction[:100]}...")
        
        print(f"\n✅ STANDARD TRAINER TEST COMPLETED SUCCESSFULLY!")
        print(f"\n💡 Ready for training with standard transformers!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()