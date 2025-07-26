"""Drug data processing from ChEMBL and other sources."""

import logging
import requests
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import time
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DrugDataProcessor:
    """Process drug data from ChEMBL and other databases."""
    
    def __init__(self, cache_dir: str = "data/cache"):
        """Initialize drug data processor."""
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.chembl_base_url = "https://www.ebi.ac.uk/chembl/api/data"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'ProteinDrugDiscovery/1.0 (research@example.com)'
        })
    
    def fetch_drug_by_chembl_id(self, chembl_id: str) -> Dict[str, Any]:
        """
        Fetch drug data from ChEMBL by ID.
        
        Args:
            chembl_id: ChEMBL compound ID
            
        Returns:
            Dictionary with drug information
        """
        try:
            # Check cache first
            cache_file = self.cache_dir / f"drug_{chembl_id}.json"
            if cache_file.exists():
                with open(cache_file, 'r') as f:
                    return json.load(f)
            
            # Fetch from ChEMBL
            url = f"{self.chembl_base_url}/molecule/{chembl_id}"
            params = {"format": "json"}
            
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            # Extract relevant information
            drug_info = self._extract_drug_info(data)
            
            # Cache the result
            with open(cache_file, 'w') as f:
                json.dump(drug_info, f, indent=2)
            
            logger.info(f"Fetched drug data for {chembl_id}")
            return drug_info
            
        except requests.RequestException as e:
            logger.error(f"Error fetching drug {chembl_id}: {e}")
            return self._create_mock_drug_data(chembl_id)
        except Exception as e:
            logger.error(f"Unexpected error for drug {chembl_id}: {e}")
            return self._create_mock_drug_data(chembl_id)
    
    def _extract_drug_info(self, chembl_data: Dict) -> Dict[str, Any]:
        """Extract relevant information from ChEMBL JSON response."""
        try:
            drug_info = {
                "chembl_id": chembl_data.get("molecule_chembl_id", ""),
                "pref_name": chembl_data.get("pref_name", ""),
                "smiles": "",
                "inchi": "",
                "molecular_weight": 0,
                "alogp": 0,
                "hbd": 0,
                "hba": 0,
                "rtb": 0,
                "psa": 0,
                "max_phase": 0,
                "indication_class": "",
                "therapeutic_flag": False
            }
            
            # SMILES
            if "molecule_structures" in chembl_data and chembl_data["molecule_structures"]:
                structures = chembl_data["molecule_structures"]
                drug_info["smiles"] = structures.get("canonical_smiles", "")
                drug_info["inchi"] = structures.get("standard_inchi", "")
            
            # Molecular properties
            if "molecule_properties" in chembl_data and chembl_data["molecule_properties"]:
                props = chembl_data["molecule_properties"]
                drug_info.update({
                    "molecular_weight": props.get("full_mwt", 0),
                    "alogp": props.get("alogp", 0),
                    "hbd": props.get("hbd", 0),
                    "hba": props.get("hba", 0),
                    "rtb": props.get("rtb", 0),
                    "psa": props.get("psa", 0)
                })
            
            # Development phase
            drug_info["max_phase"] = chembl_data.get("max_phase", 0)
            
            # Therapeutic flag
            drug_info["therapeutic_flag"] = chembl_data.get("therapeutic_flag", False)
            
            return drug_info
            
        except Exception as e:
            logger.error(f"Error extracting drug info: {e}")
            return self._create_mock_drug_data("unknown")
    
    def _create_mock_drug_data(self, chembl_id: str) -> Dict[str, Any]:
        """Create mock drug data for testing."""
        mock_drugs = {
            "CHEMBL25": {
                "pref_name": "Aspirin",
                "smiles": "CC(=O)OC1=CC=CC=C1C(=O)O",
                "molecular_weight": 180.16,
                "alogp": 1.19,
                "hbd": 1,
                "hba": 4,
                "max_phase": 4
            },
            "CHEMBL1201585": {
                "pref_name": "Ibuprofen",
                "smiles": "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
                "molecular_weight": 206.28,
                "alogp": 3.97,
                "hbd": 1,
                "hba": 2,
                "max_phase": 4
            },
            "CHEMBL112": {
                "pref_name": "Acetaminophen",
                "smiles": "CC(=O)NC1=CC=C(C=C1)O",
                "molecular_weight": 151.16,
                "alogp": 0.46,
                "hbd": 2,
                "hba": 2,
                "max_phase": 4
            }
        }
        
        if chembl_id in mock_drugs:
            data = mock_drugs[chembl_id].copy()
        else:
            data = {
                "pref_name": f"Mock Drug {chembl_id}",
                "smiles": "CCO",  # Simple ethanol as default
                "molecular_weight": 200.0,
                "alogp": 2.0,
                "hbd": 1,
                "hba": 2,
                "max_phase": 1
            }
        
        data.update({
            "chembl_id": chembl_id,
            "inchi": "",
            "rtb": 2,
            "psa": 40.0,
            "indication_class": "Mock indication",
            "therapeutic_flag": True,
            "mock": True
        })
        
        return data
    
    def search_drugs_by_target(self, target_chembl_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search drugs by target protein.
        
        Args:
            target_chembl_id: ChEMBL target ID
            limit: Maximum number of results
            
        Returns:
            List of drug information dictionaries
        """
        try:
            # Check cache
            cache_file = self.cache_dir / f"drugs_target_{target_chembl_id}_{limit}.json"
            if cache_file.exists():
                with open(cache_file, 'r') as f:
                    return json.load(f)
            
            # Search ChEMBL for activities
            url = f"{self.chembl_base_url}/activity"
            params = {
                "target_chembl_id": target_chembl_id,
                "format": "json",
                "limit": limit
            }
            
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            results = []
            
            # Extract unique compounds
            seen_compounds = set()
            for activity in data.get("activities", []):
                compound_id = activity.get("molecule_chembl_id")
                if compound_id and compound_id not in seen_compounds:
                    drug_info = self.fetch_drug_by_chembl_id(compound_id)
                    drug_info["activity_data"] = {
                        "standard_type": activity.get("standard_type"),
                        "standard_value": activity.get("standard_value"),
                        "standard_units": activity.get("standard_units"),
                        "pchembl_value": activity.get("pchembl_value")
                    }
                    results.append(drug_info)
                    seen_compounds.add(compound_id)
                    
                    if len(results) >= limit:
                        break
            
            # Cache results
            with open(cache_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            logger.info(f"Found {len(results)} drugs for target {target_chembl_id}")
            return results
            
        except Exception as e:
            logger.error(f"Error searching drugs for target {target_chembl_id}: {e}")
            return self._create_mock_target_drugs(target_chembl_id, limit)
    
    def _create_mock_target_drugs(self, target_chembl_id: str, limit: int) -> List[Dict[str, Any]]:
        """Create mock drug-target data."""
        results = []
        mock_compounds = ["CHEMBL25", "CHEMBL1201585", "CHEMBL112"]
        
        for i, compound_id in enumerate(mock_compounds[:limit]):
            drug_info = self._create_mock_drug_data(compound_id)
            drug_info["activity_data"] = {
                "standard_type": "IC50",
                "standard_value": 100.0 * (i + 1),
                "standard_units": "nM",
                "pchembl_value": 7.0 - i * 0.5
            }
            results.append(drug_info)
        
        return results
    
    def get_approved_drugs(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get approved drugs (max_phase = 4).
        
        Args:
            limit: Maximum number of drugs to retrieve
            
        Returns:
            List of approved drug information
        """
        try:
            # Search for approved drugs
            url = f"{self.chembl_base_url}/molecule"
            params = {
                "max_phase": 4,
                "format": "json",
                "limit": limit
            }
            
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            results = []
            
            for molecule in data.get("molecules", []):
                drug_info = self._extract_drug_info(molecule)
                results.append(drug_info)
            
            logger.info(f"Retrieved {len(results)} approved drugs")
            return results
            
        except Exception as e:
            logger.error(f"Error fetching approved drugs: {e}")
            return self._create_mock_approved_drugs(limit)
    
    def _create_mock_approved_drugs(self, limit: int) -> List[Dict[str, Any]]:
        """Create mock approved drug data."""
        approved_drugs = [
            "CHEMBL25",      # Aspirin
            "CHEMBL1201585", # Ibuprofen
            "CHEMBL112",     # Acetaminophen
            "CHEMBL1200766", # Atorvastatin
            "CHEMBL1200469"  # Metformin
        ]
        
        results = []
        for drug_id in approved_drugs[:limit]:
            drug_info = self._create_mock_drug_data(drug_id)
            drug_info["max_phase"] = 4
            drug_info["therapeutic_flag"] = True
            results.append(drug_info)
        
        return results
    
    def calculate_drug_likeness_score(self, drug_data: Dict[str, Any]) -> float:
        """
        Calculate drug-likeness score based on Lipinski's Rule of Five.
        
        Args:
            drug_data: Drug information dictionary
            
        Returns:
            Drug-likeness score (0-1)
        """
        try:
            score = 1.0
            
            # Molecular weight <= 500 Da
            if drug_data.get("molecular_weight", 0) > 500:
                score -= 0.25
            
            # LogP <= 5
            if drug_data.get("alogp", 0) > 5:
                score -= 0.25
            
            # Hydrogen bond donors <= 5
            if drug_data.get("hbd", 0) > 5:
                score -= 0.25
            
            # Hydrogen bond acceptors <= 10
            if drug_data.get("hba", 0) > 10:
                score -= 0.25
            
            return max(0.0, score)
            
        except Exception as e:
            logger.error(f"Error calculating drug-likeness score: {e}")
            return 0.0
    
    def get_binding_data(self, compound_chembl_id: str, target_chembl_id: str) -> List[Dict[str, Any]]:
        """
        Get binding affinity data for compound-target pair.
        
        Args:
            compound_chembl_id: ChEMBL compound ID
            target_chembl_id: ChEMBL target ID
            
        Returns:
            List of binding activity data
        """
        try:
            url = f"{self.chembl_base_url}/activity"
            params = {
                "molecule_chembl_id": compound_chembl_id,
                "target_chembl_id": target_chembl_id,
                "format": "json"
            }
            
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            activities = []
            
            for activity in data.get("activities", []):
                activity_info = {
                    "activity_id": activity.get("activity_id"),
                    "standard_type": activity.get("standard_type"),
                    "standard_value": activity.get("standard_value"),
                    "standard_units": activity.get("standard_units"),
                    "pchembl_value": activity.get("pchembl_value"),
                    "activity_comment": activity.get("activity_comment", "")
                }
                activities.append(activity_info)
            
            return activities
            
        except Exception as e:
            logger.error(f"Error fetching binding data: {e}")
            return [{
                "activity_id": "mock_activity",
                "standard_type": "IC50",
                "standard_value": 100.0,
                "standard_units": "nM",
                "pchembl_value": 7.0,
                "activity_comment": "Mock binding data"
            }]