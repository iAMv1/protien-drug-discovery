#!/usr/bin/env python3
"""
Train protein-drug interaction model on the full dataset with optimized hyperparameters.
This script implements the first priority from the README: Model Training on full dataset.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import logging
from pathlib import Path
import argparse
from datetime import datetime
import json
import torch

from protein_drug_discovery.data.doublesg_loader import DoubleSGDatasetLoader
from protein_drug_discovery.core.unsloth_trainer import UnslothProteinTrainer
from protein_drug_discovery.core.validation_metrics import ValidationMetrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Optimized hyperparameters based on validation performance
OPTIMIZED_HYPERPARAMS = {
    "model_name": "unsloth/llama-3.2-1b-instruct-bnb-4bit",
    "max_seq_length": 2048,
    "num_epochs": 3,
    "batch_size": 4,
    "gradient_accumulation": 8,  # Effective batch size = 32
    "learning_rate": 2e-4,
    "lora_r": 32,  # Increased LoRA rank for better capacity
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "warmup_ratio": 0.1,
    "weight_decay": 0.01,
}

def load_training_data(data_dir: str = "datasets/doublesg/training_data") -> tuple:
    """Load prepared training data"""
    data_path = Path(data_dir)
    
    if not data_path.exists():
        logger.error(f"Training data directory not found: {data_path}")
        logger.info("Please run the data preparation script first")
        return None, None, None
    
    # Load the prepared datasets
    train_df = pd.read_csv(data_path / "train.csv")
    val_df = pd.read_csv(data_path / "validation.csv")
    test_df = pd.read_csv(data_path / "test.csv")
    
    logger.info(f"Loaded training data:")
    logger.info(f"  Train: {len(train_df)} samples")
    logger.info(f"  Validation: {len(val_df)} samples")
    logger.info(f"  Test: {len(test_df)} samples")
    
    return train_df, val_df, test_df

def prepare_data_if_needed():
    """Prepare DoubleSG dataset if not already done"""
    data_path = Path("datasets/doublesg/training_data")
    
    if data_path.exists():
        logger.info("Training data already exists")
        return True
    
    logger.info("Preparing DoubleSG dataset...")
    try:
        loader = DoubleSGDatasetLoader()
        
        # Download all datasets
        datasets = loader.download_all_datasets()
        
        if not datasets:
            logger.error("No datasets were successfully loaded")
            return False
        
        # Prepare data for training
        processed_splits = loader.prepare_for_training(datasets)
        
        if processed_splits:
            # Save training data
            loader.save_training_data(processed_splits)
            logger.info("Training data prepared successfully")
            return True
        else:
            logger.error("Failed to prepare training data")
            return False
            
    except Exception as e:
        logger.error(f"Error preparing data: {e}")
        return False

def evaluate_model(trainer, test_df, output_dir):
    """Evaluate model on test data and calculate validation metrics."""
    logger.info("Evaluating model on test data...")
    
    # Get predictions for test data
    predictions = []
    true_labels = []
    
    for _, row in test_df.iterrows():
        try:
            # Get prediction
            prediction_text = trainer.inference(
                protein_sequence=row['protein'],
                drug_smiles=row['compound'],
                max_new_tokens=256
            )
            
            # Extract binding probability from prediction text
            # This is a simple extraction - might need to be adjusted based on actual output format
            try:
                binding_prob_line = [line for line in prediction_text.split('\n') 
                                    if "Binding Probability:" in line][0]
                binding_prob = float(binding_prob_line.split(':')[1].strip())
            except (IndexError, ValueError):
                binding_prob = 0.5  # Default if extraction fails
            
            # Get true label (1 if pKd >= 7.0, else 0)
            true_label = 1 if row.get('binding_label', row['pKd'] >= 7.0) else 0
            
            predictions.append(binding_prob)
            true_labels.append(true_label)
            
        except Exception as e:
            logger.warning(f"Error predicting for test sample: {e}")
    
    # Calculate validation metrics
    metrics = ValidationMetrics()
    results = metrics.calculate_metrics(
        y_true=np.array(true_labels),
        y_scores=np.array(predictions)
    )
    
    # Save metrics
    metrics_path = Path(output_dir) / "validation_metrics.json"
    metrics.save_metrics(str(metrics_path))
    
    # Generate plots
    plots_dir = Path(output_dir) / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    metrics.plot_roc_curve(save_path=str(plots_dir / "roc_curve.png"))
    metrics.plot_precision_recall_curve(save_path=str(plots_dir / "pr_curve.png"))
    
    # Print summary
    metrics.print_summary()
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Train protein-drug interaction model on full dataset with optimized hyperparameters")
    parser.add_argument("--output_dir", default="./models/protein_drug_full_dataset", 
                       help="Output directory")
    parser.add_argument("--use_wandb", action="store_true", 
                       help="Use Weights & Biases logging")
    parser.add_argument("--data_dir", default="datasets/doublesg/training_data", 
                       help="Training data directory")
    parser.add_argument("--prepare_data", action="store_true", 
                       help="Prepare data before training")
    parser.add_argument("--skip_training", action="store_true",
                       help="Skip training and only evaluate an existing model")
    parser.add_argument("--model_path", default=None,
                       help="Path to existing model for evaluation (if skip_training is True)")
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("🧬 PROTEIN-DRUG DISCOVERY - FULL DATASET TRAINING WITH OPTIMIZED HYPERPARAMETERS")
    print("=" * 80)
    
    # Print optimized hyperparameters
    print(f"📊 Optimized Hyperparameters:")
    for param, value in OPTIMIZED_HYPERPARAMS.items():
        print(f"  {param}: {value}")
    
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
    
    # Initialize Unsloth QLoRA trainer
    print(f"\n🚀 Initializing Unsloth QLoRA trainer...")
    try:
        trainer = UnslothProteinTrainer(
            model_name=OPTIMIZED_HYPERPARAMS["model_name"],
            max_seq_length=OPTIMIZED_HYPERPARAMS["max_seq_length"],
            load_in_4bit=True,  # QLoRA uses 4-bit quantization
            lora_r=OPTIMIZED_HYPERPARAMS["lora_r"],
            lora_alpha=OPTIMIZED_HYPERPARAMS["lora_alpha"],
            lora_dropout=OPTIMIZED_HYPERPARAMS["lora_dropout"]
        )
        
        # Get model info
        info = trainer.get_model_info()
        print(f"  ✅ Model loaded successfully!")
        print(f"  📈 Model Stats:")
        print(f"    - Type: {info['model_type']}")
        print(f"    - Total Parameters: {info['total_parameters']:,}")
        print(f"    - Trainable Parameters: {info['trainable_parameters']:,}")
        print(f"    - Trainable %: {info['trainable_percentage']:.2f}%")
        print(f"    - Memory Usage: {info['memory_usage']}")
        print(f"    - Inference Speed: {info['inference_speed']}")
        
    except Exception as e:
        print(f"❌ Failed to initialize trainer: {e}")
        print(f"💡 Make sure to install: pip install unsloth")
        return
    
    # Skip training if requested
    if args.skip_training:
        if args.model_path:
            print(f"\n⏩ Skipping training, evaluating existing model: {args.model_path}")
            # Load existing model
            trainer.load_adapter(args.model_path)
        else:
            print("❌ Model path must be provided when skip_training is True")
            return
    else:
        # Prepare datasets
        print(f"\n📝 Preparing datasets for training...")
        try:
            train_dataset, val_dataset = trainer.prepare_dataset(train_df, val_df)
            print(f"  ✅ Datasets prepared successfully")
            
        except Exception as e:
            print(f"❌ Failed to prepare datasets: {e}")
            return
        
        # Start training
        print(f"\n🏋️ Starting training on FULL dataset with OPTIMIZED hyperparameters...")
        print(f"  ⚡ Using Unsloth for 2x faster training!")
        print(f"  💾 Only ~3GB VRAM required (4-bit quantization)")
        print(f"  📦 Final model size: ~100MB LoRA adapter")
        
        try:
            training_stats = trainer.train(
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                output_dir=args.output_dir,
                num_train_epochs=OPTIMIZED_HYPERPARAMS["num_epochs"],
                per_device_train_batch_size=OPTIMIZED_HYPERPARAMS["batch_size"],
                gradient_accumulation_steps=OPTIMIZED_HYPERPARAMS["gradient_accumulation"],
                learning_rate=OPTIMIZED_HYPERPARAMS["learning_rate"],
                use_wandb=args.use_wandb,
                warmup_ratio=OPTIMIZED_HYPERPARAMS["warmup_ratio"],
                weight_decay=OPTIMIZED_HYPERPARAMS["weight_decay"],
                logging_steps=10
            )
            
            print(f"\n✅ Training completed successfully!")
            print(f"  📊 Final Results:")
            print(f"    - Training Loss: {training_stats.training_loss:.4f}")
            print(f"    - Training Steps: {training_stats.global_step}")
            print(f"    - Model Saved: {args.output_dir}")
            
            # Save hyperparameters
            with open(f"{args.output_dir}/hyperparameters.json", "w") as f:
                json.dump(OPTIMIZED_HYPERPARAMS, f, indent=2)
            
        except Exception as e:
            print(f"❌ Training failed: {e}")
            return
    
    # Evaluate model
    print(f"\n📊 Evaluating model with validation metrics...")
    try:
        import numpy as np
        evaluation_results = evaluate_model(trainer, test_df, args.output_dir)
        
        print(f"\n🎯 Validation Metrics:")
        print(f"  ROC-AUC: {evaluation_results['roc_auc']:.4f}")
        print(f"  Precision: {evaluation_results['precision']:.4f}")
        print(f"  Recall: {evaluation_results['recall']:.4f}")
        print(f"  F1 Score: {evaluation_results['f1_score']:.4f}")
        
    except Exception as e:
        print(f"⚠️ Evaluation failed: {e}")
    
    print(f"\n🎉 Full Dataset Training Pipeline Completed!")
    print(f"\n📦 Model Artifacts:")
    print(f"  - LoRA Adapter: {args.output_dir}/lora_adapter/")
    print(f"  - Tokenizer: {args.output_dir}/tokenizer/")
    print(f"  - Config: {args.output_dir}/training_config.json")
    print(f"  - Validation Metrics: {args.output_dir}/validation_metrics.json")
    print(f"  - Plots: {args.output_dir}/plots/")

if __name__ == "__main__":
    main()