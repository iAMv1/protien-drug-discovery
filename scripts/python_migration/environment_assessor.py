#!/usr/bin/env python3
"""
Environment Assessment and Backup System
Creates comprehensive backup of current Python environment before migration
"""

import os
import sys
import json
import subprocess
import winreg
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PythonInstallation:
    """Represents a Python installation on the system"""
    
    def __init__(self, version: str, path: str, executable: str):
        self.version = version
        self.path = path
        self.executable = executable
        self.packages = []
        self.in_path = False
    
    def to_dict(self):
        return {
            'version': self.version,
            'path': self.path,
            'executable': self.executable,
            'packages': self.packages,
            'in_path': self.in_path
        }

class EnvironmentAssessor:
    """Assesses current Python environment and creates backups"""
    
    def __init__(self, backup_dir: str = "migration_backup"):
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create timestamped backup subdirectory
        self.current_backup_dir = self.backup_dir / f"backup_{self.timestamp}"
        self.current_backup_dir.mkdir(exist_ok=True)
        
        logger.info(f"Backup directory: {self.current_backup_dir}")
    
    def discover_python_installations(self) -> List[PythonInstallation]:
        """Discover all Python installations on the system"""
        logger.info("Discovering Python installations...")
        installations = []
        
        # Method 1: Use py launcher to find installations
        try:
            result = subprocess.run(['py', '-0'], capture_output=True, text=True, check=True)
            lines = result.stdout.strip().split('\n')
            
            for line in lines:
                if line.strip():
                    # Parse py -0 output: " -V:3.13 *        Python 3.13 (64-bit)"
                    parts = line.split()
                    if len(parts) >= 3:
                        version_flag = parts[0]  # e.g., "-V:3.13"
                        version = version_flag.split(':')[1] if ':' in version_flag else "unknown"
                        is_default = '*' in line
                        
                        # Get full path for this version
                        try:
                            path_result = subprocess.run(
                                ['py', f'-{version}', '-c', 'import sys; print(sys.executable)'],
                                capture_output=True, text=True, check=True
                            )
                            executable = path_result.stdout.strip()
                            install_path = str(Path(executable).parent)
                            
                            installation = PythonInstallation(version, install_path, executable)
                            installations.append(installation)
                            
                            logger.info(f"Found Python {version} at {install_path} {'(default)' if is_default else ''}")
                            
                        except subprocess.CalledProcessError as e:
                            logger.warning(f"Could not get path for Python {version}: {e}")
            
        except subprocess.CalledProcessError as e:
            logger.warning(f"py launcher not available: {e}")
        
        # Method 2: Check common installation paths
        common_paths = [
            r"C:\Python*",
            r"C:\Users\*\AppData\Local\Programs\Python\Python*",
            r"C:\Program Files\Python*",
            r"C:\Program Files (x86)\Python*",
            r"D:\Python*"
        ]
        
        import glob
        for pattern in common_paths:
            for path in glob.glob(pattern):
                if os.path.isdir(path):
                    python_exe = os.path.join(path, "python.exe")
                    if os.path.exists(python_exe):
                        try:
                            result = subprocess.run(
                                [python_exe, '--version'],
                                capture_output=True, text=True, check=True
                            )
                            version = result.stdout.strip().replace('Python ', '')
                            
                            # Check if we already have this installation
                            if not any(inst.executable == python_exe for inst in installations):
                                installation = PythonInstallation(version, path, python_exe)
                                installations.append(installation)
                                logger.info(f"Found additional Python {version} at {path}")
                                
                        except subprocess.CalledProcessError:
                            continue
        
        return installations
    
    def get_installed_packages(self, python_executable: str) -> List[str]:
        """Get list of installed packages for a Python installation"""
        logger.info(f"Getting packages for {python_executable}")
        
        try:
            result = subprocess.run(
                [python_executable, '-m', 'pip', 'freeze'],
                capture_output=True, text=True, check=True
            )
            packages = [line.strip() for line in result.stdout.split('\n') if line.strip()]
            logger.info(f"Found {len(packages)} packages")
            return packages
            
        except subprocess.CalledProcessError as e:
            logger.warning(f"Could not get packages for {python_executable}: {e}")
            return []
    
    def get_system_path(self) -> str:
        """Get current system PATH variable"""
        logger.info("Getting system PATH...")
        
        try:
            # Get system PATH from registry
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, 
                              r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment") as key:
                path_value, _ = winreg.QueryValueEx(key, "PATH")
                return path_value
        except Exception as e:
            logger.warning(f"Could not read system PATH from registry: {e}")
            # Fallback to environment variable
            return os.environ.get('PATH', '')
    
    def get_user_path(self) -> str:
        """Get current user PATH variable"""
        logger.info("Getting user PATH...")
        
        try:
            # Get user PATH from registry
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Environment") as key:
                path_value, _ = winreg.QueryValueEx(key, "PATH")
                return path_value
        except Exception as e:
            logger.warning(f"Could not read user PATH from registry: {e}")
            return ""
    
    def backup_registry_entries(self) -> Dict[str, str]:
        """Backup Python-related registry entries"""
        logger.info("Backing up registry entries...")
        
        registry_backup = {}
        
        # Python-related registry keys to backup
        keys_to_backup = [
            (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Python"),
            (winreg.HKEY_CURRENT_USER, r"SOFTWARE\Python"),
            (winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment"),
            (winreg.HKEY_CURRENT_USER, r"Environment")
        ]
        
        for hkey, subkey in keys_to_backup:
            try:
                with winreg.OpenKey(hkey, subkey) as key:
                    # Get all values in this key
                    key_data = {}
                    i = 0
                    while True:
                        try:
                            name, value, type_id = winreg.EnumValue(key, i)
                            key_data[name] = {'value': value, 'type': type_id}
                            i += 1
                        except WindowsError:
                            break
                    
                    registry_backup[f"{hkey}\\{subkey}"] = key_data
                    
            except Exception as e:
                logger.warning(f"Could not backup registry key {subkey}: {e}")
        
        return registry_backup
    
    def test_current_functionality(self) -> Dict[str, bool]:
        """Test current DoubleSG-DTA functionality"""
        logger.info("Testing current project functionality...")
        
        functionality_tests = {}
        
        # Test 1: Basic imports
        try:
            import torch
            functionality_tests['torch_import'] = True
            logger.info("✓ PyTorch import successful")
        except ImportError as e:
            functionality_tests['torch_import'] = False
            logger.warning(f"✗ PyTorch import failed: {e}")
        
        # Test 2: PyTorch Geometric
        try:
            import torch_geometric
            functionality_tests['torch_geometric_import'] = True
            logger.info("✓ PyTorch Geometric import successful")
        except ImportError as e:
            functionality_tests['torch_geometric_import'] = False
            logger.warning(f"✗ PyTorch Geometric import failed: {e}")
        
        # Test 3: RDKit
        try:
            from rdkit import Chem
            functionality_tests['rdkit_import'] = True
            logger.info("✓ RDKit import successful")
        except ImportError as e:
            functionality_tests['rdkit_import'] = False
            logger.warning(f"✗ RDKit import failed: {e}")
        
        # Test 4: DoubleSG-DTA integration
        try:
            sys.path.append(os.getcwd())
            from protein_drug_discovery.models.doublesg_integration import MolecularGraphProcessor
            processor = MolecularGraphProcessor()
            graph = processor.smiles_to_graph("C")  # Simple methane molecule
            functionality_tests['doublesg_integration'] = graph['num_nodes'] > 0
            logger.info("✓ DoubleSG-DTA integration working")
        except Exception as e:
            functionality_tests['doublesg_integration'] = False
            logger.warning(f"✗ DoubleSG-DTA integration failed: {e}")
        
        # Test 5: Demo script execution
        try:
            if os.path.exists('test_doublesg_integration.py'):
                result = subprocess.run([sys.executable, 'test_doublesg_integration.py'], 
                                      capture_output=True, text=True, timeout=60)
                functionality_tests['demo_script'] = result.returncode == 0
                if result.returncode == 0:
                    logger.info("✓ Demo script executed successfully")
                else:
                    logger.warning(f"✗ Demo script failed: {result.stderr}")
            else:
                functionality_tests['demo_script'] = False
                logger.warning("✗ Demo script not found")
        except Exception as e:
            functionality_tests['demo_script'] = False
            logger.warning(f"✗ Demo script execution failed: {e}")
        
        return functionality_tests
    
    def create_environment_snapshot(self) -> Dict:
        """Create complete environment snapshot"""
        logger.info("Creating environment snapshot...")
        
        # Discover Python installations
        installations = self.discover_python_installations()
        
        # Get packages for each installation
        for installation in installations:
            installation.packages = self.get_installed_packages(installation.executable)
            # Check if this Python is in PATH
            installation.in_path = installation.path in self.get_system_path()
        
        # Create comprehensive snapshot
        snapshot = {
            'timestamp': self.timestamp,
            'python_installations': [inst.to_dict() for inst in installations],
            'system_path': self.get_system_path(),
            'user_path': self.get_user_path(),
            'current_python': {
                'executable': sys.executable,
                'version': sys.version,
                'path': sys.path
            },
            'registry_backup': self.backup_registry_entries(),
            'functionality_tests': self.test_current_functionality(),
            'environment_variables': dict(os.environ),
            'working_directory': os.getcwd()
        }
        
        return snapshot
    
    def save_snapshot(self, snapshot: Dict) -> str:
        """Save snapshot to file"""
        snapshot_file = self.current_backup_dir / "environment_snapshot.json"
        
        # Convert non-serializable objects to strings
        serializable_snapshot = self._make_serializable(snapshot)
        
        with open(snapshot_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_snapshot, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Environment snapshot saved to: {snapshot_file}")
        return str(snapshot_file)
    
    def _make_serializable(self, obj):
        """Convert objects to JSON-serializable format"""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, Path):
            return str(obj)
        elif hasattr(obj, '__dict__'):
            return self._make_serializable(obj.__dict__)
        else:
            try:
                json.dumps(obj)
                return obj
            except (TypeError, ValueError):
                return str(obj)
    
    def export_requirements_files(self, installations: List[PythonInstallation]):
        """Export requirements.txt files for each Python installation"""
        logger.info("Exporting requirements files...")
        
        for installation in installations:
            if installation.packages:
                version_safe = installation.version.replace('.', '_')
                req_file = self.current_backup_dir / f"requirements_python_{version_safe}.txt"
                
                with open(req_file, 'w') as f:
                    for package in installation.packages:
                        f.write(f"{package}\n")
                
                logger.info(f"Requirements exported for Python {installation.version}: {req_file}")
    
    def create_restoration_script(self, snapshot: Dict):
        """Create script to restore environment if needed"""
        logger.info("Creating restoration script...")
        
        script_content = f'''#!/usr/bin/env python3
"""
Environment Restoration Script
Generated on: {snapshot['timestamp']}
Use this script to restore the Python environment to its pre-migration state
"""

import os
import sys
import winreg
import subprocess
import json

def restore_system_path():
    """Restore system PATH to original state"""
    original_path = r"{snapshot['system_path']}"
    
    try:
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, 
                          r"SYSTEM\\CurrentControlSet\\Control\\Session Manager\\Environment",
                          0, winreg.KEY_SET_VALUE) as key:
            winreg.SetValueEx(key, "PATH", 0, winreg.REG_EXPAND_SZ, original_path)
        print("✓ System PATH restored")
        return True
    except Exception as e:
        print(f"✗ Failed to restore system PATH: {{e}}")
        return False

def restore_user_path():
    """Restore user PATH to original state"""
    original_path = r"{snapshot['user_path']}"
    
    if not original_path:
        return True
    
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Environment",
                          0, winreg.KEY_SET_VALUE) as key:
            winreg.SetValueEx(key, "PATH", 0, winreg.REG_EXPAND_SZ, original_path)
        print("✓ User PATH restored")
        return True
    except Exception as e:
        print(f"✗ Failed to restore user PATH: {{e}}")
        return False

def main():
    print("🔄 Starting environment restoration...")
    
    success = True
    success &= restore_system_path()
    success &= restore_user_path()
    
    if success:
        print("\\n🎉 Environment restoration completed successfully!")
        print("⚠️  Please restart your command prompt/IDE for changes to take effect")
    else:
        print("\\n❌ Environment restoration encountered errors")
        print("💡 You may need to run this script as Administrator")
    
    return success

if __name__ == "__main__":
    main()
'''
        
        script_file = self.current_backup_dir / "restore_environment.py"
        with open(script_file, 'w') as f:
            f.write(script_content)
        
        logger.info(f"Restoration script created: {script_file}")
    
    def run_complete_assessment(self) -> str:
        """Run complete environment assessment and backup"""
        logger.info("🔍 Starting complete environment assessment...")
        
        try:
            # Create environment snapshot
            snapshot = self.create_environment_snapshot()
            
            # Save snapshot to file
            snapshot_file = self.save_snapshot(snapshot)
            
            # Export requirements files
            installations = [PythonInstallation(**inst) for inst in snapshot['python_installations']]
            self.export_requirements_files(installations)
            
            # Create restoration script
            self.create_restoration_script(snapshot)
            
            # Create summary report
            self._create_summary_report(snapshot)
            
            logger.info("✅ Environment assessment completed successfully!")
            logger.info(f"📁 Backup location: {self.current_backup_dir}")
            
            return str(self.current_backup_dir)
            
        except Exception as e:
            logger.error(f"❌ Environment assessment failed: {e}")
            raise
    
    def _create_summary_report(self, snapshot: Dict):
        """Create human-readable summary report"""
        report_file = self.current_backup_dir / "assessment_report.txt"
        
        with open(report_file, 'w') as f:
            f.write("PYTHON ENVIRONMENT ASSESSMENT REPORT\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Assessment Date: {snapshot['timestamp']}\n")
            f.write(f"Working Directory: {snapshot['working_directory']}\n\n")
            
            f.write("PYTHON INSTALLATIONS FOUND:\n")
            f.write("-" * 30 + "\n")
            for inst in snapshot['python_installations']:
                f.write(f"Python {inst['version']}\n")
                f.write(f"  Location: {inst['path']}\n")
                f.write(f"  Executable: {inst['executable']}\n")
                f.write(f"  In PATH: {inst['in_path']}\n")
                f.write(f"  Packages: {len(inst['packages'])} installed\n\n")
            
            f.write("FUNCTIONALITY TESTS:\n")
            f.write("-" * 20 + "\n")
            for test, result in snapshot['functionality_tests'].items():
                status = "✓ PASS" if result else "✗ FAIL"
                f.write(f"{test}: {status}\n")
            
            f.write(f"\nCURRENT PYTHON: {snapshot['current_python']['executable']}\n")
            f.write(f"VERSION: {snapshot['current_python']['version']}\n")
        
        logger.info(f"Assessment report created: {report_file}")

def main():
    """Main function for standalone execution"""
    assessor = EnvironmentAssessor()
    backup_dir = assessor.run_complete_assessment()
    
    print(f"\n🎉 Environment assessment completed!")
    print(f"📁 Backup saved to: {backup_dir}")
    print(f"📋 Review the assessment_report.txt for details")
    print(f"🔄 Use restore_environment.py if rollback is needed")

if __name__ == "__main__":
    main()