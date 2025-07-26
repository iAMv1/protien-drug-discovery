#!/usr/bin/env python3
"""
Enhanced DoubleSG-DTA Training Script
Train the integrated DoubleSG-DTA + ESM-2 model for drug-target affinity prediction
"""

import os
import sys
import argparse
import logging
import torch
import pandas as pd
from transformers import EsmTokenizer, EsmModel

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from protein_drug_discovery.models.doublesg_integration import (
    DoubleSGDTAModel, MolecularGraphProcessor
)
from protein_drug_discovery.training.doublesg_trainer import (
    DoubleSGDTATrainer, DrugTargetAffinityDataset, create_doublesg_datasets
)
from protein_drug_discovery.data.bindingdb_processor import (
    BindingDBProcessor, create_mock_bindingdb_data
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description='Train Enhanced DoubleSG-DTA Model')
    parser.add_argument('--dataset', type=str, default='davis', 
                       choices=['davis', 'kiba', 'bindingdb', 'mock'],
                       help='Dataset to use for training')
    parser.add_argument('--data_dir', type=str, default='datasets',
                       help='Directory containing dataset files')
    parser.add_argument('--model_dir', type=str, default='models/enhanced_doublesg',
                       help='Directory to save trained models')
    parser.add_argument('--esm_model', type=str, default='facebook/esm2_t12_35M_UR50D',
                       help='ESM-2 model to use for protein encoding')
    parser.add_argument('--use_esm', action='store_true', default=True,
                       help='Use ESM-2 for protein encoding')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Training batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (cpu, cuda, auto)')
    parser.add_argument('--use_wandb', action='store_true',
                       help='Use Weights & Biases for logging')
    parser.add_argument('--create_mock_data', action='store_true',
                       help='Create mock data for testing')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    logger.info(f"Using device: {device}")
    logger.info(f"Training Enhanced DoubleSG-DTA on {args.dataset} dataset")
    
    # Create directories
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.data_dir, exist_ok=True)
    
    # Prepare dataset
    if args.create_mock_data or args.dataset == 'mock':
        logger.info("Creating mock BindingDB data for testing...")
        data_dir = create_mock_bindingdb_data(
            output_dir=os.path.join(args.data_dir, 'mock_bindingdb'),
            num_samples=1000
        )
        dataset_path = data_dir
    elif args.dataset == 'bindingdb':
        logger.info("Processing BindingDB data...")
        processor = BindingDBProcessor(output_dir=args.data_dir)
        dataset_path = processor.process_full_pipeline(max_samples=10000)
    else:
        # Use existing dataset (Davis/KIBA format)
        dataset_path = os.path.join(args.data_dir, args.dataset)
        if not os.path.exists(dataset_path):
            logger.error(f"Dataset directory not found: {dataset_path}")
            logger.info("Please provide dataset files in DoubleSG-DTA format:")
            logger.info("- train.csv")
            logger.info("- valid.csv") 
            logger.info("- test.csv")
            logger.info("Each CSV should have columns: compound_iso_smiles, target_sequence, affinity")
            return
    
    # Initialize ESM-2 model and tokenizer
    if args.use_esm:
        logger.info(f"Loading ESM-2 model: {args.esm_model}")
        esm_tokenizer = EsmTokenizer.from_pretrained(args.esm_model)
        esm_model = EsmModel.from_pretrained(args.esm_model)
        
        # Freeze ESM-2 parameters for efficiency
        for param in esm_model.parameters():
            param.requires_grad = False
    else:
        esm_tokenizer = None
        esm_model = None
    
    # Create datasets
    logger.info("Creating datasets...")
    try:
        train_dataset, val_dataset, test_dataset = create_doublesg_datasets(
            data_dir=dataset_path,
            tokenizer=esm_tokenizer
        )
        logger.info(f"Dataset sizes - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    except Exception as e:
        logger.error(f"Failed to create datasets: {e}")
        return
    
    # Initialize model
    logger.info("Initializing Enhanced DoubleSG-DTA model...")
    model = DoubleSGDTAModel(
        esm_model=esm_model,
        drug_feature_dim=78,
        esm_hidden_dim=esm_model.config.hidden_size if esm_model else 480,
        gin_hidden_dim=128,
        gin_layers=5,
        attention_heads=8,
        final_hidden_dim=256
    )
    
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    logger.info(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Initialize trainer
    logger.info("Initializing trainer...")
    trainer = DoubleSGDTATrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device=device,
        use_wandb=args.use_wandb
    )
    
    # Train model
    logger.info(f"Starting training for {args.num_epochs} epochs...")
    training_results = trainer.train(
        num_epochs=args.num_epochs,
        save_dir=args.model_dir
    )
    
    # Evaluate on test set
    logger.info("Evaluating on test set...")
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=trainer.train_loader.collate_fn,
        num_workers=0
    )
    
    # Load best model for evaluation
    best_model_path = os.path.join(args.model_dir, 'best_model.pt')
    if os.path.exists(best_model_path):
        logger.info("Loading best model for evaluation...")
        trainer.load_model(best_model_path)
    
    # Evaluate
    trainer.model.eval()
    test_predictions = []
    test_targets = []
    
    with torch.no_grad():
        for batch in test_loader:
            # Move to device
            protein_tokens = {
                k: v.to(device) for k, v in batch['protein_tokens'].items()
            }
  main()main__':
  '__name__ == f __y!")

iccessfulleted suplning cominfo("Trai    logger.s_path}")
 to: {resultlts savednfo(f"Resugger.i   
    londent=2)
 ults, f, ion.dump(res        js f:
w') ass_path, 'lt open(resu   withts.json')
 _resulingtrain 'del_dir,in(args.moath.jopath = os.pults_on
    resort js
    imp   }
    }
 n)
        est_spearma: float(tearman_r''sp      
      on),est_pears': float(tearson_r    'p
        t(test_r2),r2': floa  '         ,
 ae)st_mloat(te    'mae': f      se),
  st_me': float(te'ms            sults': {
  'test_re     ults,
 res: training__results'aining       'tr       },
 m_epochs
 nuchs': args.um_epo        'n   g_rate,
 args.learnine': at'learning_r           size,
 rgs.batch_h_size': a      'batcm,
      s.use_esm': arg'use_es       el,
     sm_moddel': args.e   'esm_mo        ': {
 igl_conf'mode   
     set,args.data'dataset':       {
    =    resultssults
 # Save re 
     ")
 :.4f}rmanst_spea R: {teSpearmant "Tesfr.info(    loggeon:.4f}")
est_pearsrson R: {t"Test Peaer.info(f")
    logg.4f}2:st_rt R²: {te(f"Teslogger.info
    ae:.4f}"){test_m"Test MAE: gger.info(f
    lo")4f}est_mse:.E: {tt MSf"Tes.info(gger")
    lo RESULTS ===NAL TEST("=== FIgger.infolo
    esultsog final r   # Lns)
    
 ictios, test_predget(test_tar = spearmanrpearman, __s)
    testionsctest_predits, tt_targepearsonr(tesn, _ = t_pearsons)
    tespredictio test__targets,_score(test2 = r2st_r    te
ions)icted test_prgets,or(test_tarrr_eabsoluten_eamae = m
    test_dictions)ts, test_pret_targetesror(quared_ere = mean_stest_ms
       
 est_targets)ray(tp.ar= nts argetest_tions)
    predictarray(test_ = np._predictions   test 
 earmanr
   spsonr, rt pears impoy.statip from scerror
   te_luabso, mean_ r2_scorerror,_en_squaredt meaetrics imporsklearn.mfrom np
    s mpy amport nurics
    iest met t # Compute
    
   )py()s.cpu().numffinitietend(a.exargetstest_t           py())
 .numu()cp().squeezeity'].inictions['afftend(predtions.extest_predic     
                a)
   ug_graph_datdrokens, in_ttedel(pror.moraineons = tdicti      pre  s
     Forward pas          # 
  
           vice)deities'].to(ch['affinnities = bat    affi   
               )device)
  to(atch'].]['bph_batch'grah['drug_ batc          
     device),_index'].to(['edge']chgraph_batrug_atch['d         b   e),
    icx'].to(devch']['ph_batug_grabatch['dr             ta = (
   _da  drug_graph          