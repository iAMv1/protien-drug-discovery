"""Launch script for FastAPI backend."""

import subprocess
import sys
import os

def main():
    """Launch the FastAPI application."""
    print("🧬 Starting Protein-Drug Discovery Platform...")
    print("🚀 Launching FastAPI Backend...")
    
    # Path to the API app
    app_path = "protein_drug_discovery.api.main:app"
    
    try:
        # Launch FastAPI with uvicorn
        subprocess.run([
            sys.executable, "-m", "uvicorn", app_path,
            "--host", "0.0.0.0",
            "--port", "8000",
            "--reload"
        ])
    except KeyboardInterrupt:
        print("\n👋 Shutting down FastAPI backend...")
    except Exception as e:
        print(f"❌ Error launching FastAPI: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()