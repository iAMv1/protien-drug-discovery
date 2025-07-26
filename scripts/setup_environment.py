#!/usr/bin/env python3
"""Environment setup script to ensure all models download to D drive."""

import os
import sys
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)

def setup_d_drive_cache():
    """Set up HuggingFace cache directories on D drive."""
    print("=" * 60)
    print("🔧 SETTING UP D DRIVE CACHE FOR HUGGINGFACE MODELS")
    print("=" * 60)
    
    # Define cache directories on D drive
    cache_dirs = {
        'HF_HOME': 'D:/huggingface_cache',
        'TRANSFORMERS_CACHE': 'D:/huggingface_cache/transformers',
        'HF_DATASETS_CACHE': 'D:/huggingface_cache/datasets',
        'HF_HUB_CACHE': 'D:/huggingface_cache/hub',
        'TORCH_HOME': 'D:/torch_cache'
    }
    
    # Create directories
    for env_var, cache_path in cache_dirs.items():
        print(f"📁 Creating directory: {cache_path}")
        Path(cache_path).mkdir(parents=True, exist_ok=True)
        
        # Set environment variable
        os.environ[env_var] = cache_path
        print(f"  ✅ Set {env_var} = {cache_path}")
    
    # Verify directories exist
    print(f"\n📋 Verifying cache directories:")
    for env_var, cache_path in cache_dirs.items():
        if Path(cache_path).exists():
            print(f"  ✅ {cache_path} - EXISTS")
        else:
            print(f"  ❌ {cache_path} - MISSING")
    
    print(f"\n💡 Environment variables set for current session.")
    print(f"💡 For permanent setup, add these to your system environment variables:")
    for env_var, cache_path in cache_dirs.items():
        print(f"  {env_var}={cache_path}")
    
    return cache_dirs

def check_dependencies():
    """Check if required dependencies are installed."""
    print(f"\n" + "=" * 60)
    print("📦 CHECKING DEPENDENCIES")
    print("=" * 60)
    
    dependencies = {
        'torch': 'PyTorch',
        'transformers': 'HuggingFace Transformers',
        'datasets': 'HuggingFace Datasets',
        'numpy': 'NumPy',
        'pandas': 'Pandas'
    }
    
    missing_deps = []
    
    for module, name in dependencies.items():
        try:
            __import__(module)
            print(f"  ✅ {name} - INSTALLED")
        except ImportError:
            print(f"  ❌ {name} - MISSING")
            missing_deps.append(module)
    
    if missing_deps:
        print(f"\n⚠️ Missing dependencies: {', '.join(missing_deps)}")
        print(f"💡 Install with: pip install {' '.join(missing_deps)}")
        return False
    else:
        print(f"\n✅ All dependencies are installed!")
        return True

def test_cache_setup():
    """Test that models will download to D drive."""
    print(f"\n" + "=" * 60)
    print("🧪 TESTING CACHE SETUP")
    print("=" * 60)
    
    try:
        from transformers import AutoTokenizer
        
        # Test with a small model
        model_name = "distilbert-base-uncased"
        cache_dir = "D:/huggingface_cache/transformers"
        
        print(f"🔄 Testing download to: {cache_dir}")
        print(f"📦 Loading small test model: {model_name}")
        
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir
        )
        
        print(f"  ✅ Model downloaded successfully to D drive!")
        
        # Check if files exist in D drive cache
        cache_path = Path(cache_dir)
        model_files = list(cache_path.glob("**/tokenizer.json"))
        
        if model_files:
            print(f"  ✅ Found model files in D drive cache")
            print(f"  📁 Cache location: {model_files[0].parent}")
        else:
            print(f"  ⚠️ Model files not found in expected location")
        
        return True
        
    except Exception as e:
        print(f"❌ Cache test failed: {e}")
        return False

def create_env_file():
    """Create a .env file with D drive cache settings."""
    print(f"\n" + "=" * 60)
    print("📝 CREATING ENVIRONMENT FILE")
    print("=" * 60)
    
    env_content = """# HuggingFace Cache Settings - D Drive
HF_HOME=D:/huggingface_cache
TRANSFORMERS_CACHE=D:/huggingface_cache/transformers
HF_DATASETS_CACHE=D:/huggingface_cache/datasets
HF_HUB_CACHE=D:/huggingface_cache/hub
TORCH_HOME=D:/torch_cache

# Project Settings
PYTHONPATH=.
"""
    
    env_file = Path(".env")
    with open(env_file, 'w') as f:
        f.write(env_content)
    
    print(f"  ✅ Created .env file with D drive cache settings")
    print(f"  📁 Location: {env_file.absolute()}")
    
    return env_file

def main():
    """Main setup function."""
    print("🚀 PROTEIN-DRUG DISCOVERY ENVIRONMENT SETUP")
    print("🎯 Ensuring all models download to D drive")
    
    # Step 1: Set up D drive cache
    cache_dirs = setup_d_drive_cache()
    
    # Step 2: Check dependencies
    deps_ok = check_dependencies()
    
    if not deps_ok:
        print(f"\n❌ Please install missing dependencies first")
        return False
    
    # Step 3: Test cache setup
    cache_ok = test_cache_setup()
    
    # Step 4: Create .env file
    env_file = create_env_file()
    
    # Summary
    print(f"\n" + "=" * 60)
    print("📋 SETUP SUMMARY")
    print("=" * 60)
    
    if cache_ok:
        print(f"✅ D drive cache setup: SUCCESS")
        print(f"✅ Dependencies check: {'SUCCESS' if deps_ok else 'FAILED'}")
        print(f"✅ Cache test: {'SUCCESS' if cache_ok else 'FAILED'}")
        print(f"✅ Environment file: CREATED")
        
        print(f"\n🎉 SETUP COMPLETED SUCCESSFULLY!")
        print(f"\n📁 All HuggingFace models will now download to:")
        for env_var, cache_path in cache_dirs.items():
            print(f"  {cache_path}")
        
        print(f"\n🚀 Ready to download ESM-2 model!")
        print(f"💡 Run: python test_real_esm.py")
        
        return True
    else:
        print(f"❌ Setup failed. Check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)