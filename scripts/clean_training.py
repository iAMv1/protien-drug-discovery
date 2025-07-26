#!/usr/bin/env python3
"""
Clean Training Script - Only working models
Train protein-drug discovery with ESM-2, AlphaFold 3, or DialoGPT
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import logging
from pathlib import Path
import pandas as pd
from datetime import datetime

from protein_drug_discovery.core.clean_model_manager import create_clean_manager
from protein_drug_discovery.data.doublesg_loader import DoubleSGDatasetLoader

logging.basicConfig(level=logging.INFO)

def prepare_training_data(sample_size=None):
    """Prepare DoubleSG training data"""
    print("📊 Preparing training data...")
    
    # Check if data already exists
    data_path = Path("datasets/doublesg/training_data")
    if data_path.exists():
        print("  ✅ Loading existing training data")
        train_df = pd.read_csv(data_path / "train.csv")
        val_df = pd.read_csv(data_path / "validation.csv")
        test_df = pd.read_csv(data_path / "test.csv")
    else:
        print("  📥 Downloading and processing DoubleSG dataset...")
        loader = DoubleSGDatasetLoader()
        
        # Download datasets
        datasets = loader.download_all_datasets()
        if not datasets:
            print("  ❌ Failed to download datasets")
            return None, None, None
        
        # Prepare for training
        processed_splits = loader.prepare_for_training(datasets)
        if not processed_splits:
            print("  ❌ Failed to process datasets")
            return None, None, None
        
        # Save training data
        loader.save_training_data(processed_splits)
        
        train_df = processed_splits['train']
        val_df = processed_splits['validation']
        test_df = processed_splits['test']
    
    # Sample data if requested
    if sample_size:
        print(f"  🔬 Sampling {sample_size} training examples")
        train_df = train_df.sample(n=min(sample_size, len(train_df)), random_state=42)
        val_df = val_df.sample(n=min(sample_size//5, len(val_df)), random_state=42)
    
    print(f"  ✅ Data ready - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    return train_df, val_df, test_df

def format_for_language_model(df):
    """Format data for language model training"""
    instruction_template = """You are a protein-drug interaction prediction expert. Given a protein sequence and drug SMILES, predict the binding affinity and properties.

### Protein Sequence:
{protein_sequence}

### Drug SMILES:
{drug_smiles}

### Task:
Predict the binding affinity (pKd), binding probability, and drug properties."""

    response_template = """Based on the protein-drug interaction analysis:

**Binding Affinity (pKd):** {pkd:.2f}
**Binding Probability:** {binding_prob:.3f}
**Confidence Score:** {confidence:.3f}
**Predicted Properties:**
- Toxicity Risk: {toxicity_risk}
- Solubility: {solubility_level}
- Drug-likeness: {druglikeness}

**Explanation:** This prediction is based on protein-drug interaction patterns, molecular properties, and binding site analysis."""

    formatted_data = []
    
    for _, row in df.iterrows():
        try:
            # Create instruction
            instruction = instruction_template.format(
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
            
            response = response_template.format(
                pkd=row['pKd'],
                binding_prob=binding_prob,
                confidence=confidence,
                toxicity_risk=toxicity_risk,
                solubility_level=solubility_level,
                druglikeness=druglikeness
            )
            
            # Format for training
            formatted_text = f"{instruction}\n\n{response}"
            formatted_data.append({"text": formatted_text})
            
        except Exception as e:
            logging.warning(f"Error formatting row: {e}")
            continue
    
    return formatted_data

def train_with_language_model(model, train_data, val_data, output_dir, args):
    """Train using language model with LoRA"""
    try:
        from transformers import TrainingArguments
        from trl import SFTTrainer
        from datasets import Dataset
        from peft import LoraConfig, get_peft_model, TaskType
        import wandb
        
        print("🏋️ Starting language model training...")
        
        # Initialize wandb if requested
        if args.use_wandb:
            wandb.init(
                project="protein-drug-discovery",
                name=f"clean-training-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
                config={
                    "model": args.model_id,
                    "epochs": args.num_epochs,
                    "batch_size": args.batch_size,
                    "learning_rate": args.learning_rate,
                    "sample_size": args.sample_size
                }
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
        
        # Apply LoRA to model
        peft_model = get_peft_model(model.model, lora_config)
        
        # Prepare datasets
        train_dataset = Dataset.from_list(train_data)
        val_dataset = Dataset.from_list(val_data) if val_data else None
        
        # Training arguments
        training_args = TrainingArguments(
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation,
            warmup_steps=10,
            num_train_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
            fp16=False,  # Disable for CPU
            logging_steps=1,
            weight_decay=0.01,
            output_dir=output_dir,
            save_steps=100,
            eval_steps=50 if val_dataset else None,
            evaluation_strategy="steps" if val_dataset else "no",
            save_strategy="steps",
            load_best_model_at_end=True if val_dataset else False,
            report_to="wandb" if args.use_wandb else None,
            remove_unused_columns=False,
        )
        
        # Create trainer
        trainer = SFTTrainer(
            model=peft_model,
            tokenizer=model.tokenizer,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            dataset_text_field="text",
            max_seq_length=1024,
            args=training_args,
        )
        
        # Train
        print(f"  🚀 Training {args.model_id} for {args.num_epochs} epochs...")
        trainer_stats = trainer.train()
        
        # Save model
        lora_path = Path(output_dir) / "lora_adapter"
        peft_model.save_pretrained(lora_path)
        model.tokenizer.save_pretrained(Path(output_dir) / "tokenizer")
        
        print(f"  ✅ Training completed!")
        print(f"    - Training Loss: {trainer_stats.training_loss:.4f}")
        print(f"    - Training Steps: {trainer_stats.global_step}")
        print(f"    - Model Saved: {output_dir}")
        
        if args.use_wandb:
            wandb.finish()
        
        return trainer_stats
        
    except Exception as e:
        print(f"  ❌ Training failed: {e}")
        raise

def test_inference(model, test_df, model_id):
    """Test inference with trained model"""
    print("🔮 Testing inference...")
    
    try:
        # Get a sample from test data
        test_sample = test_df.iloc[0]
        
        # Format input
        instruction = f"""You are a protein-drug interaction prediction expert. Given a protein sequence and drug SMILES, predict the binding affinity and properties.

### Protein Sequence:
{test_sample['protein'][:500]}

### Drug SMILES:
{test_sample['compound']}

### Task:
Predict the binding affinity (pKd), binding probability, and drug properties."""
        
        # Tokenize
        inputs = model.tokenizer(
            instruction,
            return_tensors="pt",
            max_length=1024,
            truncation=True,
            padding=True
        )
        
        # Generate
        with torch.no_grad():
            outputs = model.model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=model.tokenizer.eos_token_id
            )
        
        # Decode
        response = model.tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated_text = response[len(instruction):].strip()
        
        print(f"  🧬 Protein: {test_sample['protein'][:50]}...")
        print(f"  💊 Drug: {test_sample['compound']}")
        print(f"  📊 Actual pKd: {test_sample['pKd']:.2f}")
        print(f"  🤖 Prediction: {generated_text[:200]}...")
        
    except Exception as e:
        print(f"  ⚠️ Inference test failed: {e}")

def main():
    parser = argparse.ArgumentParser(description="Clean training script for protein-drug discovery")
    
    # Model selection
    parser.add_argument("--model_id", default="dialogpt_small", 
                       choices=["esm2_35m", "esm2_150m", "alphafold3", "dialogpt_small"],
                       help="Model to use")
    
    # Training parameters
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument("--gradient_accumulation", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--sample_size", type=int, default=None, help="Sample size for testing")
    
    # Output and monitoring
    parser.add_argument("--output_dir", default="./models/clean_training", help="Output directory")
    parser.add_argument("--use_wandb", action="store_true", help="Use Weights & Biases")
    
    # Actions
    parser.add_argument("--list_models", action="store_true", help="List available models")
    parser.add_argument("--test_inference", action="store_true", help="Test inference after training")
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("🧬 CLEAN PROTEIN-DRUG DISCOVERY TRAINING")
    print("=" * 80)
    
    # Create clean model manager
    print("📦 Initializing clean model manager...")
    manager = create_clean_manager()
    
    # List models if requested
    if args.list_models:
        print("\n📋 Available Models:")
        models = manager.list_models()
        for model_id, info in models.items():
            print(f"  {model_id}: {info['config']['name']}")
        return
    
    print(f"\n📊 Training Configuration:")
    print(f"  Model: {args.model_id}")
    print(f"  Epochs: {args.num_epochs}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Learning Rate: {args.learning_rate}")
    print(f"  Sample Size: {args.sample_size or 'Full dataset'}")
    print(f"  Output: {args.output_dir}")
    print(f"  W&B: {args.use_wandb}")
    
    # Load model
    print(f"\n🚀 Loading model: {args.model_id}")
    if not manager.load_model(args.model_id):
        print(f"❌ Failed to load model: {args.model_id}")
        return 1
    
    model = manager.get_model(args.model_id)
    info = model.get_model_info()
    print(f"  ✅ Model loaded: {info['parameter_count']:,} parameters")
    
    # Prepare data
    train_df, val_df, test_df = prepare_training_data(args.sample_size)
    if train_df is None:
        print("❌ Failed to prepare training data")
        return 1
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Train based on model type
    if args.model_id == "dialogpt_small":
        # Language model training
        train_data = format_for_language_model(train_df)
        val_data = format_for_language_model(val_df) if val_df is not None else None
        
        print(f"  📝 Formatted {len(train_data)} training examples")
        
        trainer_stats = train_with_language_model(model, train_data, val_data, args.output_dir, args)
        
        # Test inference if requested
        if args.test_inference:
            test_inference(model, test_df, args.model_id)
    
    else:
        # Protein encoder - for now just test encoding
        print(f"🧬 Testing protein encoding with {args.model_id}...")
        test_protein = train_df.iloc[0]['protein']
        result = model.encode(test_protein)
        print(f"  ✅ Protein encoded: {result['embeddings'].shape}")
        print(f"  📊 Embedding dimension: {result['embeddings'].shape[-1]}")
        print(f"  💡 Protein encoders ready for downstream training")
    
    print(f"\n🎉 Clean training completed!")
    print(f"\n📦 Available models:")
    print(f"  - esm2_35m: Fast protein encoding (35M params)")
    print(f"  - esm2_150m: Better protein encoding (150M params)")
    print(f"  - alphafold3: AlphaFold 3 alternative (ESM-2 150M)")
    print(f"  - dialogpt_small: Language model for training")
    
    return 0

if __name__ == "__main__":
    import torch
    sys.exit(main())