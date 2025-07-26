# Python Environment Setup Design Document

## Overview

This document outlines the technical design for migrating from Python 3.13 (C: drive) to Python 3.11 (D: drive) while maintaining full compatibility with the DoubleSG-DTA integration project. The design ensures a safe, reversible migration process with comprehensive testing and validation.

## Architecture

### Current State Analysis
```
Current Environment:
├── Python 3.13 (C:\Users\ItzP\AppData\Local\Programs\Python\Python313\)
│   ├── Installed packages: torch, torch-geometric, rdkit, transformers, etc.
│   ├── System PATH entries
│   └── Active development environment
├── Python 3.11 (C:\Users\ItzP\AppData\Local\Programs\Python\Python311\)
│   ├── Minimal installation
│   ├── Limited packages
│   └── Not in primary PATH
└── Project Dependencies
    ├── DoubleSG-DTA integration
    ├── Protein-drug discovery modules
    └── Demo scripts and tests
```

### Target State Design
```
Target Environment:
├── Python 3.11 (D:\Python311\)
│   ├── Complete package ecosystem
│   ├── Primary system PATH
│   ├── All DoubleSG-DTA dependencies
│   └── Optimized for development
├── Removed: Python 3.13
│   ├── Uninstalled from system
│   ├── PATH entries removed
│   └── Registry cleaned
└── Validated Project
    ├── All imports working
    ├── Demo scripts functional
    └── Test suite passing
```

## Components and Interfaces

### 1. Migration Controller
**Purpose**: Orchestrates the entire migration process with rollback capabilities.

```python
class PythonMigrationController:
    def __init__(self):
        self.backup_manager = BackupManager()
        self.python_installer = PythonInstaller()
        self.package_manager = PackageManager()
        self.path_manager = PathManager()
        self.validator = EnvironmentValidator()
    
    def execute_migration(self):
        """Execute complete migration with checkpoints"""
        try:
            self.pre_migration_backup()
            self.install_python311_to_d_drive()
            self.migrate_packages()
            self.update_system_path()
            self.remove_python313()
            self.validate_environment()
            self.cleanup_temporary_files()
        except Exception as e:
            self.rollback_migration()
            raise MigrationError(f"Migration failed: {e}")
```

### 2. Backup Manager
**Purpose**: Creates comprehensive backups before migration begins.

```python
class BackupManager:
    def create_environment_snapshot(self):
        """Create complete environment backup"""
        return {
            'python_installations': self.get_python_installations(),
            'system_path': self.get_system_path(),
            'installed_packages': self.get_all_packages(),
            'project_state': self.test_project_functionality(),
            'registry_entries': self.backup_python_registry()
        }
    
    def export_package_requirements(self, python_path):
        """Export pip freeze for specific Python installation"""
        subprocess.run([python_path, '-m', 'pip', 'freeze', '>', 'requirements_backup.txt'])
```

### 3. Python Installer
**Purpose**: Handles Python 3.11 installation to custom D: drive location.

```python
class PythonInstaller:
    def install_to_custom_location(self, target_path="D:\\Python311"):
        """Install Python 3.11 to specified location"""
        installer_url = "https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe"
        installer_path = self.download_installer(installer_url)
        
        install_command = [
            installer_path,
            '/quiet',
            f'InstallAllUsers=0',
            f'TargetDir={target_path}',
            'PrependPath=1',
            'Include_test=0',
            'Include_doc=0'
        ]
        
        return subprocess.run(install_command, check=True)
```

### 4. Package Manager
**Purpose**: Migrates all required packages to the new Python installation.

```python
class PackageManager:
    def __init__(self, new_python_path):
        self.python_exe = os.path.join(new_python_path, 'python.exe')
        self.pip_exe = os.path.join(new_python_path, 'Scripts', 'pip.exe')
    
    def install_core_packages(self):
        """Install essential packages in correct order"""
        core_packages = [
            'pip --upgrade',
            'wheel setuptools',
            'numpy',
            'torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu',
            'torch-geometric',
            'rdkit',
            'transformers',
            'pandas scikit-learn scipy tqdm',
            'jupyter notebook'
        ]
        
        for package in core_packages:
            self.install_package(package)
    
    def install_project_dependencies(self):
        """Install DoubleSG-DTA specific dependencies"""
        subprocess.run([self.pip_exe, 'install', '-r', 'requirements.txt'])
```

### 5. PATH Manager
**Purpose**: Updates system PATH variables safely.

```python
class PathManager:
    def update_system_path(self, new_python_path):
        """Update system PATH with new Python location"""
        current_path = self.get_system_path()
        
        # Remove old Python paths
        cleaned_path = self.remove_python_paths(current_path)
        
        # Add new Python paths
        new_paths = [
            new_python_path,
            os.path.join(new_python_path, 'Scripts')
        ]
        
        updated_path = ';'.join(new_paths + [cleaned_path])
        self.set_system_path(updated_path)
    
    def remove_python_paths(self, path_string):
        """Remove all Python-related paths"""
        paths = path_string.split(';')
        filtered_paths = [p for p in paths if not self.is_python_path(p)]
        return ';'.join(filtered_paths)
```

### 6. Environment Validator
**Purpose**: Comprehensive testing of the new environment.

```python
class EnvironmentValidator:
    def validate_complete_environment(self):
        """Run comprehensive validation suite"""
        results = {
            'python_version': self.check_python_version(),
            'package_imports': self.test_package_imports(),
            'doublesg_integration': self.test_doublesg_functionality(),
            'system_commands': self.test_system_commands(),
            'project_tests': self.run_project_test_suite()
        }
        
        return all(results.values())
    
    def test_doublesg_functionality(self):
        """Test DoubleSG-DTA integration specifically"""
        try:
            # Test imports
            from protein_drug_discovery.models.doublesg_integration import (
                DoubleSGDTAModel, MolecularGraphProcessor
            )
            
            # Test basic functionality
            processor = MolecularGraphProcessor()
            graph = processor.smiles_to_graph("CC(C)CC1=CC=C(C=C1)C(C)C(=O)O")
            
            return graph['num_nodes'] > 0
        except Exception as e:
            logging.error(f"DoubleSG test failed: {e}")
            return False
```

## Data Models

### Migration State Model
```python
@dataclass
class MigrationState:
    phase: str  # 'backup', 'install', 'migrate', 'validate', 'cleanup'
    status: str  # 'in_progress', 'completed', 'failed'
    timestamp: datetime
    checkpoint_data: Dict[str, Any]
    error_info: Optional[str] = None
```

### Environment Snapshot Model
```python
@dataclass
class EnvironmentSnapshot:
    python_installations: List[PythonInstallation]
    system_path: str
    installed_packages: Dict[str, List[str]]  # python_path -> package_list
    project_functionality: bool
    registry_backup: Dict[str, Any]
    creation_time: datetime
```

### Package Dependency Model
```python
@dataclass
class PackageDependency:
    name: str
    version: str
    install_command: str
    dependencies: List[str]
    validation_import: str
    critical: bool = False  # Must work for project to function
```

## Error Handling

### Migration Error Categories
1. **Installation Errors**: Python installer failures, permission issues
2. **Package Errors**: Package installation failures, dependency conflicts
3. **PATH Errors**: System PATH corruption, access denied
4. **Validation Errors**: Project functionality broken, import failures
5. **System Errors**: Registry issues, file system problems

### Error Recovery Strategy
```python
class ErrorRecoveryManager:
    def handle_migration_error(self, error_type, error_info):
        """Handle different types of migration errors"""
        recovery_strategies = {
            'installation_error': self.recover_from_installation_failure,
            'package_error': self.recover_from_package_failure,
            'path_error': self.recover_from_path_failure,
            'validation_error': self.recover_from_validation_failure,
            'system_error': self.recover_from_system_failure
        }
        
        recovery_func = recovery_strategies.get(error_type)
        if recovery_func:
            return recovery_func(error_info)
        else:
            return self.full_rollback()
```

## Testing Strategy

### Pre-Migration Testing
1. **Environment Documentation**: Capture current state completely
2. **Functionality Baseline**: Test all DoubleSG-DTA features
3. **Package Inventory**: Document all installed packages
4. **System State**: Backup PATH, registry, file locations

### Migration Testing
1. **Checkpoint Validation**: Test after each major step
2. **Rollback Testing**: Verify rollback works at each checkpoint
3. **Package Installation**: Test each package individually
4. **Integration Testing**: Test package interactions

### Post-Migration Testing
1. **System Commands**: `python --version`, `pip --version`, `py -0`
2. **Package Imports**: Test all critical package imports
3. **DoubleSG Integration**: Run complete integration test suite
4. **Demo Scripts**: Execute all demo and test scripts
5. **Performance Testing**: Ensure no performance degradation

### Automated Test Suite
```python
class MigrationTestSuite:
    def run_complete_test_suite(self):
        """Run comprehensive test suite"""
        tests = [
            self.test_python_installation,
            self.test_package_availability,
            self.test_doublesg_imports,
            self.test_molecular_processing,
            self.test_model_creation,
            self.test_demo_scripts,
            self.test_system_integration
        ]
        
        results = {}
        for test in tests:
            try:
                results[test.__name__] = test()
            except Exception as e:
                results[test.__name__] = f"FAILED: {e}"
        
        return results
```

## Implementation Timeline

### Phase 1: Preparation (30 minutes)
- Create environment snapshot
- Export package requirements
- Backup system configuration
- Download Python 3.11 installer

### Phase 2: Installation (20 minutes)
- Install Python 3.11 to D:\Python311
- Verify installation success
- Update system PATH temporarily

### Phase 3: Package Migration (45 minutes)
- Install core packages (numpy, torch, etc.)
- Install PyTorch Geometric
- Install scientific computing stack
- Install project dependencies

### Phase 4: System Configuration (15 minutes)
- Update system PATH permanently
- Remove Python 3.13 PATH entries
- Verify command line access

### Phase 5: Python 3.13 Removal (20 minutes)
- Uninstall Python 3.13 from Control Panel
- Clean up remaining files
- Remove registry entries

### Phase 6: Validation (30 minutes)
- Run complete test suite
- Validate DoubleSG-DTA functionality
- Test demo scripts
- Performance verification

### Phase 7: Cleanup (10 minutes)
- Remove temporary files
- Clean up backup files (optional)
- Document final state

**Total Estimated Time: 2.5 - 3 hours**

## Risk Mitigation

### High-Risk Scenarios
1. **System PATH Corruption**: 
   - Mitigation: Backup PATH before changes, test each modification
   - Recovery: Automated PATH restoration from backup

2. **Package Installation Failures**:
   - Mitigation: Install packages in dependency order, test each one
   - Recovery: Rollback to previous checkpoint, retry with different approach

3. **Project Functionality Broken**:
   - Mitigation: Comprehensive testing at each step
   - Recovery: Full environment rollback, investigate specific failures

### Rollback Strategy
```python
class RollbackManager:
    def execute_full_rollback(self, snapshot):
        """Complete rollback to previous state"""
        self.restore_system_path(snapshot.system_path)
        self.restore_python_installations(snapshot.python_installations)
        self.restore_registry_entries(snapshot.registry_backup)
        self.validate_rollback_success()
```

## Success Metrics

1. **Python Version**: `python --version` returns "Python 3.11.x"
2. **Installation Location**: Python located at D:\Python311\
3. **Package Availability**: All DoubleSG-DTA dependencies importable
4. **Functionality**: Demo script runs without errors
5. **Performance**: No degradation in execution speed
6. **System Integration**: All system commands work correctly

This design ensures a safe, comprehensive migration while maintaining full project functionality.