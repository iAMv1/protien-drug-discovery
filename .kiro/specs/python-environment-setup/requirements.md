# Python Environment Setup Requirements

## Introduction

This specification outlines the requirements for reconfiguring the Python development environment by moving Python 3.11 to the D: drive and removing Python 3.13 to create a clean, optimized setup for the protein-drug discovery project.

## Requirements

### Requirement 1: Python Version Management

**User Story:** As a developer, I want to have only Python 3.11 installed on my system located on the D: drive, so that I have a clean, consistent development environment with better disk space management.

#### Acceptance Criteria

1. WHEN the system is configured THEN Python 3.13 SHALL be completely uninstalled from the system
2. WHEN Python 3.11 is installed THEN it SHALL be located on the D: drive (e.g., D:\Python311\)
3. WHEN Python commands are executed THEN they SHALL use Python 3.11 from the D: drive location
4. WHEN the system PATH is checked THEN it SHALL contain only the D: drive Python 3.11 paths
5. WHEN `python --version` is executed THEN it SHALL return "Python 3.11.x"

### Requirement 2: Package Environment Migration

**User Story:** As a developer, I want all required packages for the DoubleSG-DTA integration to be properly installed in the new Python 3.11 environment, so that the project continues to work seamlessly after the migration.

#### Acceptance Criteria

1. WHEN the new Python environment is set up THEN PyTorch SHALL be installed and functional
2. WHEN the new Python environment is set up THEN PyTorch Geometric SHALL be installed and functional
3. WHEN the new Python environment is set up THEN RDKit SHALL be installed and functional
4. WHEN the new Python environment is set up THEN Transformers library SHALL be installed and functional
5. WHEN the new Python environment is set up THEN all scientific computing packages (numpy, pandas, scipy, scikit-learn) SHALL be installed
6. WHEN package imports are tested THEN all DoubleSG-DTA integration dependencies SHALL import successfully

### Requirement 3: System PATH Configuration

**User Story:** As a developer, I want the system PATH to be properly configured for the new Python location, so that Python commands work from any directory without specifying full paths.

#### Acceptance Criteria

1. WHEN the system PATH is configured THEN D:\Python311\ SHALL be in the PATH
2. WHEN the system PATH is configured THEN D:\Python311\Scripts\ SHALL be in the PATH
3. WHEN the system PATH is configured THEN old Python 3.13 paths SHALL be removed from PATH
4. WHEN `py -0` is executed THEN it SHALL show only Python 3.11 from the D: drive
5. WHEN `pip --version` is executed THEN it SHALL use pip from the D: drive Python installation

### Requirement 4: Project Compatibility Verification

**User Story:** As a developer, I want to verify that the DoubleSG-DTA integration continues to work after the Python environment migration, so that development can continue without interruption.

#### Acceptance Criteria

1. WHEN the migration is complete THEN the DoubleSG-DTA demo script SHALL run successfully
2. WHEN the migration is complete THEN all import statements in the project SHALL work correctly
3. WHEN the migration is complete THEN PyTorch operations SHALL execute without errors
4. WHEN the migration is complete THEN molecular graph processing SHALL function correctly
5. WHEN the migration is complete THEN the test suite SHALL pass all tests

### Requirement 5: Backup and Recovery Plan

**User Story:** As a developer, I want a backup and recovery plan in case the migration fails, so that I can restore the working environment if needed.

#### Acceptance Criteria

1. WHEN the migration begins THEN a list of currently installed packages SHALL be exported
2. WHEN the migration begins THEN current PATH configuration SHALL be backed up
3. WHEN the migration begins THEN project dependencies SHALL be documented
4. WHEN migration fails THEN the system SHALL be restorable to the previous working state
5. WHEN recovery is needed THEN clear rollback instructions SHALL be available

## Migration Steps Overview

1. **Pre-migration Assessment**
   - Document current Python installations
   - Export package lists from both Python versions
   - Backup system PATH configuration
   - Test current DoubleSG-DTA functionality

2. **Python 3.11 Reinstallation**
   - Download Python 3.11 installer
   - Install to D:\Python311\ with custom options
   - Configure system PATH variables
   - Verify installation

3. **Python 3.13 Removal**
   - Uninstall Python 3.13 from Control Panel
   - Clean up remaining files and registry entries
   - Remove Python 3.13 paths from system PATH
   - Verify complete removal

4. **Package Migration**
   - Install pip and essential tools
   - Install PyTorch with CPU support
   - Install PyTorch Geometric
   - Install scientific computing stack
   - Install project-specific dependencies

5. **Verification and Testing**
   - Test Python installation
   - Verify all package imports
   - Run DoubleSG-DTA integration tests
   - Validate project functionality

## Risk Assessment

- **High Risk**: System PATH corruption could break other applications
- **Medium Risk**: Package compatibility issues with new Python location
- **Low Risk**: Project code changes needed for new environment

## Success Criteria

The migration is considered successful when:
- Only Python 3.11 is installed on the D: drive
- All DoubleSG-DTA dependencies are functional
- The demo script runs without errors
- System performance is maintained or improved