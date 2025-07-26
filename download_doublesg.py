"""
Download and analyze DoubleSG-DTA repository for real model implementation
"""

import requests
import json
import os
from pathlib import Path
import zipfile
import io

def download_doublesg_repo():
    """Download DoubleSG-DTA repository"""
    
    print("Downloading DoubleSG-DTA repository...")
    
    # Download repository as ZIP
    repo_url = "https://github.com/YongtaoQian/DoubleSG-DTA/archive/refs/heads/main.zip"
    
    try:
        response = requests.get(repo_url, timeout=30)
        response.raise_for_status()
        
        # Extract ZIP
        with zipfile.ZipFile(io.BytesIO(response.content)) as zip_file:
            zip_file.extractall("temp_doublesg")
        
        print("Repository downloaded successfully")
        return True
        
    except Exception as e:
        print(f"Failed to download repository: {e}")
        return False

def analyze_repo_structure():
    """Analyze the repository structure"""
    
    print("\nRepository Structure:")
    
    repo_path = Path("temp_doublesg/DoubleSG-DTA-main")
    if not repo_path.exists():
        print("Repository not found")
        return
    
    for root, dirs, files in os.walk(repo_path):
        level = root.replace(str(repo_path), '').count(os.sep)
        indent = ' ' * 2 * level
        print(f"{indent}{os.path.basename(root)}/")
        
        subindent = ' ' * 2 * (level + 1)
        for file in files:
            print(f"{subindent}{file}")

def read_key_files():
    """Read key implementation files"""
    
    repo_path = Path("temp_doublesg/DoubleSG-DTA-main")
    
    key_files = [
        "create_data.py",
        "utils.py", 
        "training.py",
        "ginconv.py",
        "DoubleSG-DTA_Train_main.py",
        "load_data.py"
    ]
    
    print("\nKey Files Analysis:")
    
    for filename in key_files:
        file_path = repo_path / filename
        if file_path.exists():
            print(f"\n--- {filename} ---")
            try:
                content = file_path.read_text(encoding='utf-8')
                # Show first 500 characters
                print(content[:500] + "..." if len(content) > 500 else content)
            except Exception as e:
                print(f"Error reading {filename}: {e}")
        else:
            print(f"{filename} not found")

def analyze_data_format():
    """Analyze the data format used"""
    
    repo_path = Path("temp_doublesg/DoubleSG-DTA-main")
    
    # Look for data files
    data_dirs = ["data", "dataset", "datasets"]
    
    print("\nData Format Analysis:")
    
    for data_dir in data_dirs:
        data_path = repo_path / data_dir
        if data_path.exists():
            print(f"\nFound data directory: {data_dir}")
            for file in data_path.iterdir():
                if file.is_file():
                    print(f"  {file.name}")
                    
                    # Try to read first few lines
                    try:
                        if file.suffix in ['.txt', '.csv', '.tsv']:
                            lines = file.read_text(encoding='utf-8').split('\n')[:5]
                            for i, line in enumerate(lines):
                                if line.strip():
                                    print(f"    Line {i+1}: {line[:100]}...")
                    except Exception as e:
                        print(f"    Error reading: {e}")

if __name__ == "__main__":
    print("DoubleSG-DTA Repository Analysis")
    print("=" * 50)
    
    # Download repository
    if download_doublesg_repo():
        # Analyze structure
        analyze_repo_structure()
        
        # Read key files
        read_key_files()
        
        # Analyze data format
        analyze_data_format()
        
        print("\nAnalysis complete!")
    else:
        print("Failed to download repository")