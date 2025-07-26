# scripts/train_standard_model.py

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import logging
from pathlib import Path
import argparse
from datetime import datetime

from protein_drug_discovery.data.doublesg_loader import DoubleSGDatasetLoader
from protein_drug_discovery.core.standard_trainer import StandardProteinTrainer

logging.basicConfig(level=logging.INFO)

def load_training_data(data_dir: str = "datasets/doublesg/training_data") -> tuple:
    """Load prepared training data"""
    data_path = Path(data_dir)
    
    if not data_path.exists():
        logging.error(f"Training data directory not found: {data_path}")
        logging.info("Please run the data preparation script first")
        return None, None, None
    
    # Load the prepared datasets
    train_df = pd.read_csv(data_path / "train.csv")
    val_df = pd.read_csv(data_path / "validation.csv")
    test_df = pd.read_csv(data_path / "test.csv")
    
    logging.info(f"Loaded training data:")
    logging.info(f"  Train: {len(train_df)} samples")
    logging.info(f"  Validation: {len(val_df)} samples")
    logging.info(f"  Test: {len(test_df)} samples")
    
    return train_df, val_df, test_df

def prepare_data_if_needed():
    """Prepare DoubleSG dataset if not already done"""
    data_path = Path("datasets/doublesg/training_data")
    
    if data_path.exists():
        logging.info("Training data already exists")
        return True
    
    logging.info("Preparing DoubleSG dataset...")
    try:
        loader = DoubleSGDatasetLoader()
        
        # Download all datasets
        datasets = loader.download_all_datasets()
        
        if not datasets:
            logging.error("No datasets were successfully loaded")
            return False
        
        # Prepare data for training
        processed_splits = loader.prepare_for_training(datasets)
        
        if processed_splits:
            # Save training data
            loader.save_training_data(processed_splits)
            logging.info("Training data prepared successfully")
            return True
        else:
            logging.error("Failed to prepare training data")
            return False
            
    except Exception as e:
        logging.error(f"Error preparing data: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Train protein-drug interaction model with Standard QLoRA")
    parser.add_argument("--model_name", default="microsoft/DialoGPT-small", 
                       help="HuggingFace model name")
    parser.add_argument("--max_seq_length", type=int, default=1024, 
                       help="Maximum sequence length")
    parser.add_argument("--num_epochs", type=int, default=1, 
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=2, 
                       help="Training batch size")
    parser.add_argument("--gradient_accumulation", type=int, default=4, 
                       help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=2e-4, 
                       help="Learning rate")
    parser.add_argument("--output_dir", default="./models/protein_drug_standard", 
                       help="Output directory")
    parser.add_argument("--use_wandb", action="store_true", 
                       help="Use Weights & Biases logging")
    parser.add_argument("--data_dir", default="datasets/doublesg/training_data", 
                       help="Training data directory")
    parser.add_argument("--sample_size", type=int, default=None, 
                       help="Use only a sample of the data for testing")
    parser.add_argument("--prepare_data", action="store_true", 
                       help="Prepare data before training")
    parser.add_argument("--load_in_4bit", action="store_true", 
                       help="Use 4-bit quantization (requires bitsandbytes)")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🧬 PROTEIN-DRUG DISCOVERY - STANDARD QLORA TRAINING")
    print("=" * 60)
    print(f"📊 Configuration:")
    print(f"  Model: {args.model_name}")
    print(f"  Output: {args.output_dir}")
    print(f"  Epochs: {args.num_epochs}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Gradient Accumulation: {args.gradient_accumulation}")
    print(f"  Effective Batch Size: {args.batch_size * args.gradient_accumulation}")
    print(f"  Learning Rate: {args.learning_rate}")
    print(f"  Max Sequence Length: {args.max_seq_length}")
    print(f"  4-bit Quantization: {args.load_in_4bit}")
    
    # Prepare data if requested
    if args.prepare_data:
        print(f"\n📥 Preparing DoubleSG dataset...")
        if not prepare_data_if_needed():
            print("❌ Failed to prepare data")
            return
    
    # Load training data
    print(f"\n📊 Loading training data...")
    train_df, val_df, test_df = load_training_data(args.data_dir)
    
    if train_df is None:
        print("❌ Failed to load training data")
        print("💡 Try running with --prepare_data flag")
        return
    
    # Sample data if requested (for testing)
    if args.sample_size:
        print(f"🔬 Using sample of {args.sample_size} training examples")
        train_df = train_df.sample(n=min(args.sample_size, len(train_df)), random_state=42)
        val_df = val_df.sample(n=min(args.sample_size//5, len(val_df)), random_state=42)
    
    # Initialize Standard QLoRA trainer
    print(f"\n🚀 Initializing Standard QLoRA trainer...")
    try:
        trainer = StandardProteinTrainer(
            model_name=args.model_name,
            max_seq_length=args.max_seq_length,
            load_in_4bit=args.load_in_4bit
        )
        
        # Get model info
        info = trainer.get_model_info()
        print(f"  ✅ Model loaded successfully!")
        print(f"  📈 Model Stats:")
        print(f"    - Type: {info['model_type']}")
        print(f"    - Total Parameters: {info['total_parameters']:,}")
        print(f"    - Trainable Parameters: {info['trainable_parameters']:,}")
        print(f"    - Trainable %: {info['trainable_percentage']:.2f}%")
        print(f"    - 4-bit Quantization: {info['load_in_4bit']}")
        
    except Exception as e:
        print(f"❌ Failed to initialize trainer: {e}")
        print(f"💡 Make sure to install: pip install transformers trl peft")
        if args.load_in_4bit:
            print(f"💡 For 4-bit quantization: pip install bitsandbytes")
        return
    
    # Prepare datasets
    print(f"\n📝 Preparing datasets for training...")
    try:
        train_dataset, val_dataset = trainer.prepare_dataset(train_df, val_df)
        print(f"  ✅ Datasets prepared successfully")
        
    except Exception as e:
        print(f"❌ Failed to prepare datasets: {e}")
        return
    
    # Start training
    print(f"\n🏋️ Starting Standard QLoRA training...")
    print(f"  🔧 Using standard transformers library")
    if args.load_in_4bit:
        print(f"  💾 4-bit quantization enabled for memory efficiency")
    
    try:
        training_stats = trainer.train(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            output_dir=args.output_dir,
            num_train_epochs=args.num_epochs,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation,
            learning_rate=args.learning_rate,
            use_wandb=args.use_wandb,
            warmup_steps=10,
            logging_steps=1
        )
        
        print(f"\n✅ Training completed successfully!")
        print(f"  📊 Final Results:")
        print(f"    - Training Loss: {training_stats.training_loss:.4f}")
        print(f"    - Training Steps: {training_stats.global_step}")
        print(f"    - Model Saved: {args.output_dir}")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test inference
    print(f"\n🔮 Testing inference...")
    try:
        # Get a sample from test data
        test_sample = test_df.iloc[0]
        
        prediction = trainer.inference(
            protein_sequence=test_sample['protein'],
            drug_smiles=test_sample['compound'],
            max_new_tokens=128
        )
        
        print(f"\n📋 Test Prediction:")
        print(f"  🧬 Protein: {test_sample['protein'][:50]}...")
        print(f"  💊 Drug: {test_sample['compound']}")
        print(f"  📊 Actual pKd: {test_sample['pKd']:.2f}")
        print(f"  🤖 Prediction:")
        print(f"    {prediction}")
        
    except Exception as e:
        print(f"⚠️ Inference test failed: {e}")
    
    print(f"\n🎉 Standard QLoRA Training Pipeline Completed!")
    print(f"\n💡 Next Steps:")
    print(f"  1. Evaluate model performance on full test set")
    print(f"  2. Deploy LoRA adapter for inference")
    print(f"  3. Integrate with FastAPI backend")
    print(f"  4. Create Streamlit UI for predictions")
    
    print(f"\n📦 Model Artifacts:")
    print(f"  - LoRA Adapter: {args.output_dir}/lora_adapter/")
    print(f"  - Tokenizer: {args.output_dir}/tokenizer/")
    print(f"  - Config: {args.output_dir}/training_config.json")

if __name__ == "__main__":
    main()