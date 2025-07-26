#!/usr/bin/env python3
"""
Package Inventory System
Detailed analysis and export of Python packages from multiple installations
"""

import os
import sys
import json
import subprocess
import pkg_resources
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass, asdict
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class PackageInfo:
    """Detailed information about an installed package"""
    name: str
    version: str
    location: str
    requires: List[str]
    required_by: List[str]
    installer: str = "unknown"
    size: int = 0
    editable: bool = False

class PackageInventoryManager:
    """Manages detailed package inventory across Python installations"""
    
    def __init__(self, output_dir: str = "migration_backup"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def get_detailed_package_info(self, python_executable: str) -> List[PackageInfo]:
        """Get detailed information about all packages in a Python installation"""
        logger.info(f"Analyzing packages for {python_executable}")
        
        packages = []
        
        try:
            # Get pip list with detailed information
            result = subprocess.run([
                python_executable, '-m', 'pip', 'list', '--format=json'
            ], capture_output=True, text=True, check=True)
            
            pip_packages = json.loads(result.stdout)
            
            for pkg_data in pip_packages:
                try:
                    # Get additional package information using pip show
                    show_result = subprocess.run([
                        python_executable, '-m', 'pip', 'show', pkg_data['name']
                    ], capture_output=True, text=True, check=True)
                    
                    # Parse pip show output
                    show_info = self._parse_pip_show_output(show_result.stdout)
                    
                    package_info = PackageInfo(
                        name=pkg_data['name'],
                        version=pkg_data['version'],
                        location=show_info.get('Location', ''),
                        requires=show_info.get('Requires', '').split(', ') if show_info.get('Requires') else [],
                        required_by=show_info.get('Required-by', '').split(', ') if show_info.get('Required-by') else [],
                        installer=show_info.get('Installer', 'unknown'),
                        editable=show_info.get('Editable project location') is not None
                    )
                    
                    # Calculate package size if possible
                    if package_info.location:
                        package_info.size = self._calculate_package_size(package_info.location, package_info.name)
                    
                    packages.append(package_info)
                    
                except subprocess.CalledProcessError as e:
                    logger.warning(f"Could not get details for package {pkg_data['name']}: {e}")
                    # Create basic package info
                    packages.append(PackageInfo(
                        name=pkg_data['name'],
                        version=pkg_data['version'],
                        location='',
                        requires=[],
                        required_by=[]
                    ))
            
            logger.info(f"Analyzed {len(packages)} packages")
            return packages
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to get package list for {python_executable}: {e}")
            return []
    
    def _parse_pip_show_output(self, output: str) -> Dict[str, str]:
        """Parse pip show command output"""
        info = {}
        for line in output.split('\n'):
            if ':' in line:
                key, value = line.split(':', 1)
                info[key.strip()] = value.strip()
        return info
    
    def _calculate_package_size(self, location: str, package_name: str) -> int:
        """Calculate approximate size of a package"""
        try:
            package_path = Path(location) / package_name
            if package_path.exists():
                total_size = 0
                for file_path in package_path.rglob('*'):
                    if file_path.is_file():
                        total_size += file_path.stat().st_size
                return total_size
        except Exception as e:
            logger.debug(f"Could not calculate size for {package_name}: {e}")
        return 0
    
    def analyze_package_dependencies(self, packages: List[PackageInfo]) -> Dict[str, Dict]:
        """Analyze package dependency relationships"""
        logger.info("Analyzing package dependencies...")
        
        # Create package lookup
        package_lookup = {pkg.name.lower(): pkg for pkg in packages}
        
        dependency_analysis = {
            'total_packages': len(packages),
            'dependency_tree': {},
            'orphaned_packages': [],
            'critical_packages': [],
            'large_packages': [],
            'editable_packages': []
        }
        
        # Build dependency tree
        for package in packages:
            deps = []
            for req in package.requires:
                if req and req.lower() in package_lookup:
                    deps.append({
                        'name': req,
                        'version': package_lookup[req.lower()].version
                    })
            
            dependency_analysis['dependency_tree'][package.name] = {
                'version': package.version,
                'dependencies': deps,
                'dependents': package.required_by,
                'size': package.size
            }
        
        # Find orphaned packages (no dependents)
        for package in packages:
            if not package.required_by or (len(package.required_by) == 1 and package.required_by[0] == ''):
                dependency_analysis['orphaned_packages'].append({
                    'name': package.name,
                    'version': package.version,
                    'size': package.size
                })
        
        # Find critical packages (many dependents)
        for package in packages:
            if len(package.required_by) > 5:
                dependency_analysis['critical_packages'].append({
                    'name': package.name,
                    'version': package.version,
                    'dependents_count': len(package.required_by),
                    'dependents': package.required_by
                })
        
        # Find large packages (>10MB)
        for package in packages:
            if package.size > 10 * 1024 * 1024:  # 10MB
                dependency_analysis['large_packages'].append({
                    'name': package.name,
                    'version': package.version,
                    'size': package.size,
                    'size_mb': round(package.size / (1024 * 1024), 2)
                })
        
        # Find editable packages
        for package in packages:
            if package.editable:
                dependency_analysis['editable_packages'].append({
                    'name': package.name,
                    'version': package.version,
                    'location': package.location
                })
        
        return dependency_analysis
    
    def create_migration_requirements(self, packages: List[PackageInfo], 
                                    python_version: str) -> Dict[str, List[str]]:
        """Create categorized requirements for migration"""
        logger.info(f"Creating migration requirements for Python {python_version}")
        
        # Critical packages for DoubleSG-DTA
        critical_packages = {
            'torch', 'torchvision', 'torchaudio', 'torch-geometric',
            'rdkit', 'transformers', 'numpy', 'pandas', 'scipy',
            'scikit-learn', 'matplotlib', 'seaborn', 'jupyter',
            'notebook', 'tqdm', 'requests'
        }
        
        # Development packages
        dev_packages = {
            'pytest', 'black', 'flake8', 'mypy', 'ipython',
            'jupyterlab', 'pre-commit', 'sphinx'
        }
        
        requirements = {
            'critical': [],
            'development': [],
            'optional': [],
            'system': []
        }
        
        for package in packages:
            package_spec = f"{package.name}=={package.version}"
            
            if package.name.lower() in critical_packages:
                requirements['critical'].append(package_spec)
            elif package.name.lower() in dev_packages:
                requirements['development'].append(package_spec)
            elif package.name.lower() in {'pip', 'setuptools', 'wheel'}:
                requirements['system'].append(package_spec)
            else:
                requirements['optional'].append(package_spec)
        
        return requirements
    
    def export_package_inventory(self, python_installations: List[Dict]) -> str:
        """Export complete package inventory for all Python installations"""
        logger.info("Exporting complete package inventory...")
        
        inventory_dir = self.output_dir / f"package_inventory_{self.timestamp}"
        inventory_dir.mkdir(exist_ok=True)
        
        complete_inventory = {
            'timestamp': self.timestamp,
            'installations': {}
        }
        
        for installation in python_installations:
            version = installation['version']
            executable = installation['executable']
            
            logger.info(f"Processing Python {version}...")
            
            # Get detailed package information
            packages = self.get_detailed_package_info(executable)
            
            # Analyze dependencies
            dependency_analysis = self.analyze_package_dependencies(packages)
            
            # Create migration requirements
            migration_requirements = self.create_migration_requirements(packages, version)
            
            # Store in inventory
            complete_inventory['installations'][version] = {
                'executable': executable,
                'path': installation['path'],
                'packages': [asdict(pkg) for pkg in packages],
                'dependency_analysis': dependency_analysis,
                'migration_requirements': migration_requirements
            }
            
            # Export individual requirements files
            version_safe = version.replace('.', '_')
            
            # Critical packages (must install)
            critical_file = inventory_dir / f"requirements_critical_python_{version_safe}.txt"
            with open(critical_file, 'w') as f:
                for req in migration_requirements['critical']:
                    f.write(f"{req}\n")
            
            # Development packages (optional)
            dev_file = inventory_dir / f"requirements_dev_python_{version_safe}.txt"
            with open(dev_file, 'w') as f:
                for req in migration_requirements['development']:
                    f.write(f"{req}\n")
            
            # All packages (complete backup)
            all_file = inventory_dir / f"requirements_all_python_{version_safe}.txt"
            with open(all_file, 'w') as f:
                for category in migration_requirements.values():
                    for req in category:
                        f.write(f"{req}\n")
            
            logger.info(f"Requirements exported for Python {version}")
        
        # Save complete inventory
        inventory_file = inventory_dir / "complete_package_inventory.json"
        with open(inventory_file, 'w') as f:
            json.dump(complete_inventory, f, indent=2)
        
        # Create summary report
        self._create_inventory_report(complete_inventory, inventory_dir)
        
        logger.info(f"Package inventory exported to: {inventory_dir}")
        return str(inventory_dir)
    
    def _create_inventory_report(self, inventory: Dict, output_dir: Path):
        """Create human-readable inventory report"""
        report_file = output_dir / "package_inventory_report.txt"
        
        with open(report_file, 'w') as f:
            f.write("PYTHON PACKAGE INVENTORY REPORT\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Generated: {inventory['timestamp']}\n\n")
            
            for version, data in inventory['installations'].items():
                f.write(f"PYTHON {version}\n")
                f.write("-" * 20 + "\n")
                f.write(f"Location: {data['path']}\n")
                f.write(f"Executable: {data['executable']}\n")
                f.write(f"Total Packages: {data['dependency_analysis']['total_packages']}\n")
                
                # Critical packages
                critical_count = len(data['migration_requirements']['critical'])
                f.write(f"Critical Packages: {critical_count}\n")
                
                # Large packages
                large_packages = data['dependency_analysis']['large_packages']
                if large_packages:
                    f.write(f"Large Packages (>10MB): {len(large_packages)}\n")
                    for pkg in large_packages[:5]:  # Top 5
                        f.write(f"  - {pkg['name']} ({pkg['size_mb']} MB)\n")
                
                # Orphaned packages
                orphaned = data['dependency_analysis']['orphaned_packages']
                f.write(f"Orphaned Packages: {len(orphaned)}\n")
                
                # Editable packages
                editable = data['dependency_analysis']['editable_packages']
                if editable:
                    f.write(f"Editable Packages: {len(editable)}\n")
                    for pkg in editable:
                        f.write(f"  - {pkg['name']} at {pkg['location']}\n")
                
                f.write("\n")
        
        logger.info(f"Inventory report created: {report_file}")

def main():
    """Main function for standalone execution"""
    # This would typically be called by the environment assessor
    # but can be run standalone for testing
    
    inventory_manager = PackageInventoryManager()
    
    # Mock installation data for testing
    mock_installations = [
        {
            'version': '3.13',
            'executable': sys.executable,
            'path': str(Path(sys.executable).parent)
        }
    ]
    
    inventory_dir = inventory_manager.export_package_inventory(mock_installations)
    print(f"Package inventory exported to: {inventory_dir}")

if __name__ == "__main__":
    main()