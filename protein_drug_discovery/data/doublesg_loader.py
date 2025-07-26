# protein_drug_discovery/data/doublesg_loader.py

import requests
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Optional, Dict, List, Tuple
import json
import pickle

logging.basicConfig(level=logging.INFO)

class DoubleSGDatasetLoader:
    """Loader for DoubleSG-DTA dataset from GitHub repository"""
    
    def __init__(self, cache_dir: str = "./datasets/doublesg"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.base_url = "https://raw.githubusercontent.com/YongtaoQian/DoubleSG-DTA/main/data/"
        
    def download_all_datasets(self) -> Dict[str, Dict[str, pd.DataFrame]]:
        """Download all available datasets from DoubleSG-DTA repository"""
        datasets = {}
        
        # Try to download Davis, KIBA, and BindingDB datasets
        dataset_names = ['davis', 'kiba', 'bindingdb']
        
        for dataset_name in dataset_names:
            logging.info(f"Loading {dataset_name} dataset...")
            dataset_data = self._load_dataset(dataset_name)
            if dataset_data:
                datasets[dataset_name] = dataset_data
                
        return datasets
    
    def _load_dataset(self, dataset_name: str) -> Optional[Dict[str, pd.DataFrame]]:
        """Load a specific dataset (davis, kiba, or bindingdb)"""
        dataset_splits = {}
        
        # Common file patterns to try
        file_patterns = [
            f"{dataset_name}/train.csv",
            f"{dataset_name}/test.csv", 
            f"{dataset_name}/valid.csv",
            f"{dataset_name}/validation.csv",
            f"{dataset_name}/train_fold_0.csv",
            f"{dataset_name}/test_fold_0.csv",
            f"{dataset_name}/ligands_can.txt",
            f"{dataset_name}/proteins.txt",
            f"{dataset_name}/Y.txt",
            f"{dataset_name}/affinity.txt"
        ]
        
        for file_pattern in file_patterns:
            success = self._download_file(file_pattern)
            if success:
                split_name = self._extract_split_name(file_pattern)
                df = self._load_file(file_pattern)
                if df is not None:
                    dataset_splits[split_name] = df
        
        # If we got ligands and proteins files, create interaction pairs
        if 'ligands' in dataset_splits and 'proteins' in dataset_splits:
            interaction_df = self._create_interaction_pairs(dataset_splits['ligands'], dataset_splits['proteins'], dataset_name)
            if interaction_df is not None:
                dataset_splits['interactions'] = interaction_df
        
        # If we got traditional format files (ligands, proteins, affinity matrix)
        if self._has_traditional_format(dataset_name):
            traditional_df = self._process_traditional_format(dataset_name)
            if traditional_df is not None:
                dataset_splits['full'] = traditional_df
        
        return dataset_splits if dataset_splits else None
    
    def _download_file(self, file_path: str) -> bool:
        """Download a single file from the repository"""
        local_file = self.cache_dir / file_path.replace('/', '_')
        
        if local_file.exists():
            return True
            
        url = f"{self.base_url}{file_path}"
        
        try:
            logging.info(f"Downloading {url}")
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            local_file.parent.mkdir(parents=True, exist_ok=True)
            local_file.write_bytes(response.content)
            
            logging.info(f"Successfully downloaded {file_path}")
            return True
            
        except requests.exceptions.RequestException as e:
            logging.warning(f"Could not download {url}: {e}")
            return False
    
    def _extract_split_name(self, file_path: str) -> str:
        """Extract split name from file path"""
        filename = Path(file_path).stem
        
        if 'train' in filename:
            return 'train'
        elif 'test' in filename:
            return 'test'
        elif 'valid' in filename or 'validation' in filename:
            return 'valid'
        elif 'ligands' in filename:
            return 'ligands'
        elif 'proteins' in filename:
            return 'proteins'
        elif filename in ['Y', 'affinity']:
            return 'affinity'
        else:
            return filename
    
    def _load_file(self, file_path: str) -> Optional[pd.DataFrame]:
        """Load a downloaded file into a DataFrame"""
        local_file = self.cache_dir / file_path.replace('/', '_')
        
        if not local_file.exists():
            return None
            
        try:
            if local_file.suffix == '.csv':
                df = pd.read_csv(local_file)
                logging.info(f"Loaded {file_path}: {len(df)} rows")
                return df
            elif local_file.suffix == '.txt':
                content = local_file.read_text().strip()
                
                # Check if it's JSON format
                if content.startswith('{') and content.endswith('}'):
                    try:
                        data_dict = json.loads(content)
                        
                        # Convert dictionary to DataFrame
                        if 'ligands' in file_path or 'compounds' in file_path:
                            # This is compound data (ID -> SMILES)
                            df = pd.DataFrame([
                                {'compound_id': k, 'compound': v} 
                                for k, v in data_dict.items()
                            ])
                        elif 'proteins' in file_path:
                            # This is protein data (ID -> sequence)
                            df = pd.DataFrame([
                                {'protein_id': k, 'protein': v} 
                                for k, v in data_dict.items()
                            ])
                        else:
                            # Generic key-value data
                            df = pd.DataFrame([
                                {'id': k, 'data': v} 
                                for k, v in data_dict.items()
                            ])
                        
                        logging.info(f"Loaded JSON {file_path}: {len(df)} entries")
                        return df
                        
                    except json.JSONDecodeError:
                        logging.warning(f"Failed to parse JSON in {file_path}, trying as text")
                
                # Handle as regular text file
                lines = content.split('\n')
                
                # Check if it's a matrix (multiple values per line)
                if len(lines) > 0 and len(lines[0].split()) > 1:
                    # It's a matrix
                    matrix_data = []
                    for line in lines:
                        try:
                            row = [float(x) for x in line.split()]
                            matrix_data.append(row)
                        except ValueError:
                            continue
                    df = pd.DataFrame(matrix_data)
                else:
                    # It's a list
                    df = pd.DataFrame({'data': lines})
                
                logging.info(f"Loaded text {file_path}: {len(df)} rows")
                return df
                
        except Exception as e:
            logging.error(f"Error loading {file_path}: {e}")
            return None
    
    def _create_interaction_pairs(self, ligands_df: pd.DataFrame, proteins_df: pd.DataFrame, dataset_name: str) -> Optional[pd.DataFrame]:
        """Create interaction pairs from separate ligands and proteins DataFrames"""
        try:
            # Generate synthetic affinity values for demonstration
            # In a real scenario, you would have actual affinity data
            records = []
            
            # Create a subset of interactions (not all combinations to keep it manageable)
            max_interactions = 10000  # Limit to prevent memory issues
            
            ligand_sample = ligands_df.sample(min(len(ligands_df), 100), random_state=42)
            protein_sample = proteins_df.sample(min(len(proteins_df), 100), random_state=42)
            
            interaction_count = 0
            for _, ligand_row in ligand_sample.iterrows():
                for _, protein_row in protein_sample.iterrows():
                    if interaction_count >= max_interactions:
                        break
                    
                    # Generate synthetic affinity (this would be real data in practice)
                    # Using a random value between 4.0 and 10.0 (typical pKd range)
                    synthetic_affinity = np.random.uniform(4.0, 10.0)
                    
                    records.append({
                        'compound': ligand_row['compound'],
                        'protein': protein_row['protein'],
                        'affinity': synthetic_affinity,
                        'compound_id': ligand_row['compound_id'],
                        'protein_id': protein_row['protein_id'],
                        'dataset': dataset_name,
                        'synthetic': True  # Flag to indicate synthetic data
                    })
                    
                    interaction_count += 1
                
                if interaction_count >= max_interactions:
                    break
            
            if records:
                df = pd.DataFrame(records)
                logging.info(f"Created {len(df)} synthetic interaction pairs for {dataset_name}")
                return df
            else:
                logging.warning(f"No interaction pairs created for {dataset_name}")
                return None
                
        except Exception as e:
            logging.error(f"Error creating interaction pairs for {dataset_name}: {e}")
            return None

    def _has_traditional_format(self, dataset_name: str) -> bool:
        """Check if dataset has traditional format files"""
        required_files = [
            f"{dataset_name}_ligands_can.txt",
            f"{dataset_name}_proteins.txt", 
            f"{dataset_name}_Y.txt"
        ]
        
        return all((self.cache_dir / f).exists() for f in required_files)
    
    def _process_traditional_format(self, dataset_name: str) -> Optional[pd.DataFrame]:
        """Process traditional format (ligands, proteins, affinity matrix)"""
        try:
            ligands_file = self.cache_dir / f"{dataset_name}_ligands_can.txt"
            proteins_file = self.cache_dir / f"{dataset_name}_proteins.txt"
            affinity_file = self.cache_dir / f"{dataset_name}_Y.txt"
            
            # Load data
            ligands = ligands_file.read_text().strip().split('\n')
            proteins = proteins_file.read_text().strip().split('\n')
            
            # Load affinity matrix
            affinity_lines = affinity_file.read_text().strip().split('\n')
            affinity_matrix = []
            for line in affinity_lines:
                row = [float(x) for x in line.split()]
                affinity_matrix.append(row)
            affinity_matrix = np.array(affinity_matrix)
            
            # Create interaction records
            records = []
            for i, smiles in enumerate(ligands):
                for j, sequence in enumerate(proteins):
                    if i < affinity_matrix.shape[0] and j < affinity_matrix.shape[1]:
                        affinity_value = affinity_matrix[i, j]
                        
                        # Skip invalid affinities
                        if np.isnan(affinity_value) or affinity_value <= 0:
                            continue
                            
                        records.append({
                            'compound': smiles,
                            'protein': sequence,
                            'affinity': affinity_value,
                            'compound_id': f"{dataset_name.upper()}_{i:04d}",
                            'protein_id': f"{dataset_name.upper()}_PROT_{j:04d}",
                            'dataset': dataset_name
                        })
            
            df = pd.DataFrame(records)
            logging.info(f"Processed traditional format for {dataset_name}: {len(df)} interactions")
            return df
            
        except Exception as e:
            logging.error(f"Error processing traditional format for {dataset_name}: {e}")
            return None
    
    def get_dataset_statistics(self, datasets: Dict[str, Dict[str, pd.DataFrame]]) -> Dict:
        """Get statistics about loaded datasets"""
        stats = {}
        
        for dataset_name, splits in datasets.items():
            dataset_stats = {
                'splits': list(splits.keys()),
                'total_interactions': 0,
                'unique_compounds': set(),
                'unique_proteins': set()
            }
            
            for split_name, df in splits.items():
                dataset_stats['total_interactions'] += len(df)
                
                if 'compound' in df.columns:
                    dataset_stats['unique_compounds'].update(df['compound'].unique())
                if 'protein' in df.columns:
                    dataset_stats['unique_proteins'].update(df['protein'].unique())
            
            dataset_stats['unique_compounds'] = len(dataset_stats['unique_compounds'])
            dataset_stats['unique_proteins'] = len(dataset_stats['unique_proteins'])
            
            stats[dataset_name] = dataset_stats
        
        return stats
    
    def create_training_splits(self, datasets: Dict[str, Dict[str, pd.DataFrame]], 
                             split_ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1)) -> Dict[str, pd.DataFrame]:
        """Create train/validation/test splits from loaded datasets"""
        
        # Combine all datasets
        all_records = []
        for dataset_name, splits in datasets.items():
            for split_name, df in splits.items():
                df_copy = df.copy()
                df_copy['source_dataset'] = dataset_name
                df_copy['source_split'] = split_name
                all_records.append(df_copy)
        
        if not all_records:
            logging.error("No data available for splitting")
            return {}
        
        combined_df = pd.concat(all_records, ignore_index=True)
        
        # Ensure required columns exist
        required_columns = ['compound', 'protein', 'affinity']
        missing_columns = [col for col in required_columns if col not in combined_df.columns]
        
        if missing_columns:
            logging.error(f"Missing required columns: {missing_columns}")
            return {}
        
        # Remove duplicates based on compound-protein pairs
        combined_df = combined_df.drop_duplicates(subset=['compound', 'protein'])
        
        # Shuffle the data
        combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Create splits
        n_total = len(combined_df)
        n_train = int(n_total * split_ratios[0])
        n_val = int(n_total * split_ratios[1])
        
        train_df = combined_df[:n_train]
        val_df = combined_df[n_train:n_train + n_val]
        test_df = combined_df[n_train + n_val:]
        
        splits = {
            'train': train_df,
            'validation': val_df,
            'test': test_df
        }
        
        # Log split statistics
        for split_name, df in splits.items():
            logging.info(f"{split_name.capitalize()} split: {len(df)} samples")
        
        return splits
    
    def prepare_for_training(self, datasets: Dict[str, Dict[str, pd.DataFrame]]) -> Dict[str, pd.DataFrame]:
        """Prepare datasets for model training with proper formatting"""
        
        # Create training splits
        splits = self.create_training_splits(datasets)
        
        if not splits:
            return {}
        
        # Process each split for training
        processed_splits = {}
        
        for split_name, df in splits.items():
            # Validate and clean protein sequences
            df = df[df['protein'].str.len() > 10]  # Minimum sequence length
            df = df[df['protein'].str.len() < 1024]  # Maximum sequence length for ESM-2
            
            # Validate SMILES strings (basic validation)
            df = df[df['compound'].str.len() > 5]  # Minimum SMILES length
            df = df[df['compound'].str.len() < 200]  # Maximum SMILES length
            
            # Convert affinity values to appropriate format
            if 'affinity' in df.columns:
                # Convert to pKd if needed (assuming input might be Kd in nM)
                df['pKd'] = df['affinity'].apply(lambda x: -np.log10(x * 1e-9) if x > 100 else x)
                
                # Filter out extreme values
                df = df[(df['pKd'] >= 4.0) & (df['pKd'] <= 12.0)]
            
            # Add binary binding labels (for classification tasks)
            if 'pKd' in df.columns:
                df['binding_label'] = (df['pKd'] >= 6.0).astype(int)  # Threshold for binding
            
            # Reset index
            df = df.reset_index(drop=True)
            
            processed_splits[split_name] = df
            logging.info(f"Processed {split_name}: {len(df)} samples after filtering")
        
        return processed_splits
    
    def save_training_data(self, processed_splits: Dict[str, pd.DataFrame], 
                          output_dir: str = "training_data"):
        """Save processed training data in formats suitable for model training"""
        
        output_path = self.cache_dir / output_dir
        output_path.mkdir(exist_ok=True)
        
        for split_name, df in processed_splits.items():
            # Save as CSV
            csv_path = output_path / f"{split_name}.csv"
            df.to_csv(csv_path, index=False)
            
            # Save as pickle for faster loading
            pkl_path = output_path / f"{split_name}.pkl"
            df.to_pickle(pkl_path)
            
            # Create separate files for sequences and labels (for easier model loading)
            sequences_path = output_path / f"{split_name}_sequences.txt"
            with open(sequences_path, 'w') as f:
                for _, row in df.iterrows():
                    f.write(f"{row['protein']}\t{row['compound']}\n")
            
            labels_path = output_path / f"{split_name}_labels.txt"
            with open(labels_path, 'w') as f:
                for _, row in df.iterrows():
                    f.write(f"{row['pKd']}\t{row['binding_label']}\n")
            
            logging.info(f"Saved {split_name} data to {output_path}")
        
        # Save dataset metadata
        metadata = {
            'total_samples': sum(len(df) for df in processed_splits.values()),
            'splits': {name: len(df) for name, df in processed_splits.items()},
            'features': list(processed_splits['train'].columns) if 'train' in processed_splits else [],
            'created_at': pd.Timestamp.now().isoformat()
        }
        
        metadata_path = output_path / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logging.info(f"Training data preparation complete. Saved to {output_path}")
        return output_path
    
    def save_processed_data(self, datasets: Dict[str, Dict[str, pd.DataFrame]], 
                          output_file: str = "doublesg_processed.pkl"):
        """Save processed datasets to pickle file"""
        output_path = self.cache_dir / output_file
        
        with open(output_path, 'wb') as f:
            pickle.dump(datasets, f)
        
        logging.info(f"Saved processed datasets to {output_path}")
    
    def load_processed_data(self, input_file: str = "doublesg_processed.pkl") -> Dict:
        """Load processed datasets from pickle file"""
        input_path = self.cache_dir / input_file
        
        if not input_path.exists():
            logging.error(f"Processed data file not found: {input_path}")
            return {}
        
        with open(input_path, 'rb') as f:
            datasets = pickle.load(f)
        
        logging.info(f"Loaded processed datasets from {input_path}")
        return datasets