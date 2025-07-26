#!/usr/bin/env python3
"""
System PATH Management and Backup
Handles Windows system PATH backup, modification, and restoration
"""

import os
import winreg
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import logging
from datetime import datetime
import subprocess

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PathManager:
    """Manages Windows system PATH with backup and restoration capabilities"""
    
    def __init__(self, backup_dir: str = "migration_backup"):
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def get_system_path(self) -> str:
        """Get current system PATH from registry"""
        try:
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, 
                              r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment") as key:
                path_value, _ = winreg.QueryValueEx(key, "PATH")
                logger.info("Successfully read system PATH from registry")
                return path_value
        except Exception as e:
            logger.error(f"Failed to read system PATH: {e}")
            raise
    
    def get_user_path(self) -> str:
        """Get current user PATH from registry"""
        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Environment") as key:
                try:
                    path_value, _ = winreg.QueryValueEx(key, "PATH")
                    logger.info("Successfully read user PATH from registry")
                    return path_value
                except FileNotFoundError:
                    logger.info("No user PATH variable found")
                    return ""
        except Exception as e:
            logger.error(f"Failed to read user PATH: {e}")
            return ""
    
    def parse_path_entries(self, path_string: str) -> List[str]:
        """Parse PATH string into individual entries"""
        if not path_string:
            return []
        
        # Split by semicolon and clean up entries
        entries = []
        for entry in path_string.split(';'):
            entry = entry.strip()
            if entry:  # Skip empty entries
                entries.append(entry)
        
        return entries
    
    def identify_python_paths(self, path_entries: List[str]) -> Dict[str, List[str]]:
        """Identify Python-related paths in PATH entries"""
        python_paths = {
            'python_installations': [],
            'python_scripts': [],
            'other_python': []
        }
        
        for entry in path_entries:
            entry_lower = entry.lower()
            
            # Check for Python installation directories
            if 'python' in entry_lower:
                if entry_lower.endswith('scripts') or '\\scripts' in entry_lower:
                    python_paths['python_scripts'].append(entry)
                elif any(pattern in entry_lower for pattern in ['python3', 'python2', 'python\\', 'python.exe']):
                    python_paths['python_installations'].append(entry)
                else:
                    python_paths['other_python'].append(entry)
        
        return python_paths
    
    def create_path_backup(self) -> Dict[str, any]:
        """Create comprehensive PATH backup"""
        logger.info("Creating PATH backup...")
        
        # Get current PATH values
        system_path = self.get_system_path()
        user_path = self.get_user_path()
        
        # Parse PATH entries
        system_entries = self.parse_path_entries(system_path)
        user_entries = self.parse_path_entries(user_path)
        
        # Identify Python-related paths
        system_python_paths = self.identify_python_paths(system_entries)
        user_python_paths = self.identify_python_paths(user_entries)
        
        # Create backup data structure
        backup_data = {
            'timestamp': self.timestamp,
            'system_path': {
                'raw': system_path,
                'entries': system_entries,
                'python_paths': system_python_paths
            },
            'user_path': {
                'raw': user_path,
                'entries': user_entries,
                'python_paths': user_python_paths
            },
            'combined_entries': system_entries + user_entries,
            'environment_path': os.environ.get('PATH', ''),
            'python_executable_paths': self._find_python_executables()
        }
        
        # Save backup to file
        backup_file = self.backup_dir / f"path_backup_{self.timestamp}.json"
        with open(backup_file, 'w') as f:
            json.dump(backup_data, f, indent=2)
        
        logger.info(f"PATH backup saved to: {backup_file}")
        return backup_data
    
    def _find_python_executables(self) -> List[Dict[str, str]]:
        """Find all Python executables accessible from PATH"""
        executables = []
        
        # Use 'where' command to find Python executables
        commands_to_check = ['python', 'python3', 'py']
        
        for cmd in commands_to_check:
            try:
                result = subprocess.run(['where', cmd], capture_output=True, text=True, check=True)
                paths = result.stdout.strip().split('\n')
                
                for path in paths:
                    if path.strip():
                        try:
                            # Get version information
                            version_result = subprocess.run([path, '--version'], 
                                                          capture_output=True, text=True, check=True)
                            version = version_result.stdout.strip() or version_result.stderr.strip()
                            
                            executables.append({
                                'command': cmd,
                                'path': path.strip(),
                                'version': version
                            })
                        except subprocess.CalledProcessError:
                            executables.append({
                                'command': cmd,
                                'path': path.strip(),
                                'version': 'unknown'
                            })
            
            except subprocess.CalledProcessError:
                # Command not found
                continue
        
        return executables
    
    def validate_path_modification(self, new_path: str) -> Dict[str, any]:
        """Validate a proposed PATH modification"""
        logger.info("Validating PATH modification...")
        
        validation_result = {
            'valid': True,
            'warnings': [],
            'errors': [],
            'analysis': {}
        }
        
        # Parse new PATH
        new_entries = self.parse_path_entries(new_path)
        
        # Check for duplicates
        seen_entries = set()
        duplicates = []
        for entry in new_entries:
            entry_normalized = entry.lower()
            if entry_normalized in seen_entries:
                duplicates.append(entry)
            seen_entries.add(entry_normalized)
        
        if duplicates:
            validation_result['warnings'].append(f"Duplicate PATH entries found: {duplicates}")
        
        # Check for non-existent directories
        non_existent = []
        for entry in new_entries:
            if not os.path.exists(entry):
                non_existent.append(entry)
        
        if non_existent:
            validation_result['warnings'].append(f"Non-existent directories in PATH: {non_existent}")
        
        # Check PATH length (Windows has limits)
        path_length = len(new_path)
        if path_length > 2048:
            validation_result['errors'].append(f"PATH too long ({path_length} chars, max 2048)")
            validation_result['valid'] = False
        elif path_length > 1024:
            validation_result['warnings'].append(f"PATH is quite long ({path_length} chars)")
        
        # Analyze Python paths
        python_paths = self.identify_python_paths(new_entries)
        validation_result['analysis']['python_paths'] = python_paths
        
        # Check for potential conflicts
        if len(python_paths['python_installations']) > 1:
            validation_result['warnings'].append("Multiple Python installations in PATH may cause conflicts")
        
        return validation_result
    
    def set_system_path(self, new_path: str, validate: bool = True) -> bool:
        """Set system PATH with optional validation"""
        if validate:
            validation = self.validate_path_modification(new_path)
            if not validation['valid']:
                logger.error(f"PATH validation failed: {validation['errors']}")
                return False
            
            if validation['warnings']:
                logger.warning(f"PATH validation warnings: {validation['warnings']}")
        
        try:
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, 
                              r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment",
                              0, winreg.KEY_SET_VALUE) as key:
                winreg.SetValueEx(key, "PATH", 0, winreg.REG_EXPAND_SZ, new_path)
            
            logger.info("System PATH updated successfully")
            
            # Notify system of environment change
            self._notify_environment_change()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to set system PATH: {e}")
            return False
    
    def set_user_path(self, new_path: str, validate: bool = True) -> bool:
        """Set user PATH with optional validation"""
        if validate:
            validation = self.validate_path_modification(new_path)
            if not validation['valid']:
                logger.error(f"PATH validation failed: {validation['errors']}")
                return False
        
        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Environment",
                              0, winreg.KEY_SET_VALUE) as key:
                if new_path:
                    winreg.SetValueEx(key, "PATH", 0, winreg.REG_EXPAND_SZ, new_path)
                else:
                    # Delete user PATH if empty
                    try:
                        winreg.DeleteValue(key, "PATH")
                    except FileNotFoundError:
                        pass  # Already doesn't exist
            
            logger.info("User PATH updated successfully")
            self._notify_environment_change()
            return True
            
        except Exception as e:
            logger.error(f"Failed to set user PATH: {e}")
            return False
    
    def _notify_environment_change(self):
        """Notify Windows of environment variable changes"""
        try:
            import win32gui
            import win32con
            
            # Broadcast WM_SETTINGCHANGE message
            win32gui.SendMessage(win32con.HWND_BROADCAST, win32con.WM_SETTINGCHANGE, 0, 'Environment')
            logger.info("Environment change notification sent")
        except ImportError:
            logger.warning("win32gui not available, environment change notification skipped")
        except Exception as e:
            logger.warning(f"Failed to send environment change notification: {e}")
    
    def remove_python_paths(self, path_string: str, python_versions_to_remove: List[str] = None) -> str:
        """Remove Python-related paths from PATH string"""
        logger.info("Removing Python paths from PATH...")
        
        entries = self.parse_path_entries(path_string)
        python_paths = self.identify_python_paths(entries)
        
        # Collect all Python paths to remove
        paths_to_remove = set()
        
        if python_versions_to_remove:
            # Remove specific Python versions
            for entry in entries:
                for version in python_versions_to_remove:
                    if version in entry.lower():
                        paths_to_remove.add(entry)
        else:
            # Remove all Python paths
            for category in python_paths.values():
                paths_to_remove.update(category)
        
        # Filter out Python paths
        cleaned_entries = [entry for entry in entries if entry not in paths_to_remove]
        
        logger.info(f"Removed {len(paths_to_remove)} Python-related PATH entries")
        for removed_path in paths_to_remove:
            logger.info(f"  Removed: {removed_path}")
        
        return ';'.join(cleaned_entries)
    
    def add_python_paths(self, path_string: str, python_paths: List[str], position: str = 'beginning') -> str:
        """Add Python paths to PATH string"""
        logger.info(f"Adding Python paths to PATH at {position}...")
        
        entries = self.parse_path_entries(path_string)
        
        # Validate new paths
        for path in python_paths:
            if not os.path.exists(path):
                logger.warning(f"Path does not exist: {path}")
        
        if position == 'beginning':
            new_entries = python_paths + entries
        else:  # 'end'
            new_entries = entries + python_paths
        
        logger.info(f"Added {len(python_paths)} Python paths to PATH")
        for added_path in python_paths:
            logger.info(f"  Added: {added_path}")
        
        return ';'.join(new_entries)
    
    def restore_path_from_backup(self, backup_file: str) -> bool:
        """Restore PATH from backup file"""
        logger.info(f"Restoring PATH from backup: {backup_file}")
        
        try:
            with open(backup_file, 'r') as f:
                backup_data = json.load(f)
            
            # Restore system PATH
            system_success = self.set_system_path(backup_data['system_path']['raw'], validate=False)
            
            # Restore user PATH
            user_success = self.set_user_path(backup_data['user_path']['raw'], validate=False)
            
            if system_success and user_success:
                logger.info("PATH restoration completed successfully")
                return True
            else:
                logger.error("PATH restoration encountered errors")
                return False
                
        except Exception as e:
            logger.error(f"Failed to restore PATH from backup: {e}")
            return False
    
    def create_path_report(self, backup_data: Dict) -> str:
        """Create human-readable PATH analysis report"""
        report_file = self.backup_dir / f"path_analysis_report_{self.timestamp}.txt"
        
        with open(report_file, 'w') as f:
            f.write("SYSTEM PATH ANALYSIS REPORT\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Generated: {backup_data['timestamp']}\n\n")
            
            # System PATH analysis
            f.write("SYSTEM PATH:\n")
            f.write("-" * 15 + "\n")
            f.write(f"Total entries: {len(backup_data['system_path']['entries'])}\n")
            f.write(f"Length: {len(backup_data['system_path']['raw'])} characters\n\n")
            
            # Python paths in system PATH
            sys_python = backup_data['system_path']['python_paths']
            f.write("Python-related paths in system PATH:\n")
            for category, paths in sys_python.items():
                if paths:
                    f.write(f"  {category}: {len(paths)}\n")
                    for path in paths:
                        f.write(f"    - {path}\n")
            f.write("\n")
            
            # User PATH analysis
            f.write("USER PATH:\n")
            f.write("-" * 10 + "\n")
            f.write(f"Total entries: {len(backup_data['user_path']['entries'])}\n")
            f.write(f"Length: {len(backup_data['user_path']['raw'])} characters\n\n")
            
            # Python paths in user PATH
            user_python = backup_data['user_path']['python_paths']
            f.write("Python-related paths in user PATH:\n")
            for category, paths in user_python.items():
                if paths:
                    f.write(f"  {category}: {len(paths)}\n")
                    for path in paths:
                        f.write(f"    - {path}\n")
            f.write("\n")
            
            # Python executables found
            f.write("PYTHON EXECUTABLES FOUND:\n")
            f.write("-" * 25 + "\n")
            for exe in backup_data['python_executable_paths']:
                f.write(f"Command: {exe['command']}\n")
                f.write(f"  Path: {exe['path']}\n")
                f.write(f"  Version: {exe['version']}\n\n")
        
        logger.info(f"PATH analysis report created: {report_file}")
        return str(report_file)

def main():
    """Main function for standalone execution"""
    path_manager = PathManager()
    
    # Create PATH backup
    backup_data = path_manager.create_path_backup()
    
    # Create analysis report
    report_file = path_manager.create_path_report(backup_data)
    
    print(f"PATH backup and analysis completed!")
    print(f"Report: {report_file}")

if __name__ == "__main__":
    main()