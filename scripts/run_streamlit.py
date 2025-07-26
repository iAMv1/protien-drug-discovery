"""Launch script for Streamlit UI."""

import subprocess
import sys
import os

def main():
    """Launch the Streamlit application."""
    print("🧬 Starting Protein-Drug Discovery Platform...")
    print("📊 Launching Streamlit UI...")
    
    # Get the correct path to the Streamlit app
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    app_path = os.path.join(project_root, "protein_drug_discovery", "ui", "streamlit_app.py")
    
    print(f"🔍 Looking for app at: {app_path}")
    
    if not os.path.exists(app_path):
        print(f"❌ Error: {app_path} not found!")
        print(f"📁 Current directory: {os.getcwd()}")
        print(f"📁 Script directory: {script_dir}")
        print(f"📁 Project root: {project_root}")
        sys.exit(1)
    
    try:
        # Launch Streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", app_path,
            "--server.port", "8501",
            "--server.address", "localhost"
        ])
    except KeyboardInterrupt:
        print("\n👋 Shutting down Streamlit UI...")
    except Exception as e:
        print(f"❌ Error launching Streamlit: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()