"""Drug molecule processing using RDKit for SMILES and molecular descriptors."""

import logging
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DrugProcessor:
    """Process drug molecules from SMILES strings and calculate descriptors."""
    
    def __init__(self):
        """Initialize drug processor with RDKit."""
        self.rdkit_available = False
        try:
            from rdkit import Chem
            from rdkit.Chem import Descriptors, rdMolDescriptors
            self.Chem = Chem
            self.Descriptors = Descriptors
            self.rdMolDescriptors = rdMolDescriptors
            self.rdkit_available = True
            logger.info("RDKit loaded successfully")
        except ImportError:
            logger.warning("RDKit not available, using mock implementation")
            self._setup_mock_rdkit()
    
    def _setup_mock_rdkit(self):
        """Setup mock RDKit for testing when RDKit is not available."""
        class MockMol:
            def __init__(self, smiles):
                self.smiles = smiles
        
        class MockChem:
            @staticmethod
            def MolFromSmiles(smiles):
                if smiles and isinstance(smiles, str) and len(smiles) > 0:
                    return MockMol(smiles)
                return None
        
        class MockDescriptors:
            @staticmethod
            def MolWt(mol):
                return len(mol.smiles) * 10  # Mock molecular weight
            
            @staticmethod
            def MolLogP(mol):
                return len(mol.smiles) * 0.1  # Mock LogP
            
            @staticmethod
            def NumHDonors(mol):
                return mol.smiles.count('O') + mol.smiles.count('N')
            
            @staticmethod
            def NumHAcceptors(mol):
                return mol.smiles.count('O') + mol.smiles.count('N') * 2
            
            @staticmethod
            def TPSA(mol):
                return len(mol.smiles) * 2  # Mock TPSA
        
        class MockRdMolDescriptors:
            @staticmethod
            def CalcNumRotatableBonds(mol):
                return mol.smiles.count('-')
        
        self.Chem = MockChem()
        self.Descriptors = MockDescriptors()
        self.rdMolDescriptors = MockRdMolDescriptors()
    
    def process_smiles(self, smiles: str) -> Dict[str, Any]:
        """
        Process SMILES string and extract molecular descriptors.
        
        Args:
            smiles: SMILES string representation of molecule
            
        Returns:
            Dictionary with molecular descriptors and properties
        """
        try:
            if not smiles or not isinstance(smiles, str):
                raise ValueError("Invalid SMILES string")
            
            # Parse SMILES
            mol = self.Chem.MolFromSmiles(smiles)
            if mol is None:
                raise ValueError(f"Could not parse SMILES: {smiles}")
            
            # Calculate molecular descriptors
            descriptors = self._calculate_descriptors(mol)
            
            # Calculate drug-likeness properties
            drug_likeness = self._calculate_drug_likeness(descriptors)
            
            # Create molecular fingerprint
            fingerprint = self._calculate_fingerprint(mol, smiles)
            
            return {
                "smiles": smiles,
                "descriptors": descriptors,
                "drug_likeness": drug_likeness,
                "fingerprint": fingerprint,
                "valid": True
            }
            
        except Exception as e:
            logger.error(f"Error processing SMILES {smiles}: {e}")
            return {
                "smiles": smiles,
                "descriptors": {},
                "drug_likeness": {},
                "fingerprint": [],
                "valid": False,
                "error": str(e)
            }
    
    def _calculate_descriptors(self, mol) -> Dict[str, float]:
        """Calculate molecular descriptors."""
        try:
            descriptors = {
                "molecular_weight": self.Descriptors.MolWt(mol),
                "logp": self.Descriptors.MolLogP(mol),
                "hbd": self.Descriptors.NumHDonors(mol),  # Hydrogen bond donors
                "hba": self.Descriptors.NumHAcceptors(mol),  # Hydrogen bond acceptors
                "tpsa": self.Descriptors.TPSA(mol),  # Topological polar surface area
                "rotatable_bonds": self.rdMolDescriptors.CalcNumRotatableBonds(mol)
            }
            
            # Add more descriptors if RDKit is available
            if self.rdkit_available:
                try:
                    descriptors.update({
                        "num_rings": self.rdMolDescriptors.CalcNumRings(mol),
                        "num_aromatic_rings": self.rdMolDescriptors.CalcNumAromaticRings(mol),
                        "fraction_csp3": self.rdMolDescriptors.CalcFractionCsp3(mol),
                        "num_heteroatoms": self.rdMolDescriptors.CalcNumHeteroatoms(mol)
                    })
                except:
                    pass  # Skip if descriptor calculation fails
            
            return descriptors
            
        except Exception as e:
            logger.error(f"Error calculating descriptors: {e}")
            return {}
    
    def _calculate_drug_likeness(self, descriptors: Dict[str, float]) -> Dict[str, Any]:
        """Calculate drug-likeness properties (Lipinski's Rule of Five, etc.)."""
        try:
            # Lipinski's Rule of Five
            lipinski_violations = 0
            lipinski_rules = {}
            
            # Rule 1: Molecular weight <= 500 Da
            mw_ok = descriptors.get("molecular_weight", 0) <= 500
            lipinski_rules["molecular_weight_ok"] = mw_ok
            if not mw_ok:
                lipinski_violations += 1
            
            # Rule 2: LogP <= 5
            logp_ok = descriptors.get("logp", 0) <= 5
            lipinski_rules["logp_ok"] = logp_ok
            if not logp_ok:
                lipinski_violations += 1
            
            # Rule 3: Hydrogen bond donors <= 5
            hbd_ok = descriptors.get("hbd", 0) <= 5
            lipinski_rules["hbd_ok"] = hbd_ok
            if not hbd_ok:
                lipinski_violations += 1
            
            # Rule 4: Hydrogen bond acceptors <= 10
            hba_ok = descriptors.get("hba", 0) <= 10
            lipinski_rules["hba_ok"] = hba_ok
            if not hba_ok:
                lipinski_violations += 1
            
            # Veber's rules (additional drug-likeness criteria)
            veber_rules = {
                "rotatable_bonds_ok": descriptors.get("rotatable_bonds", 0) <= 10,
                "tpsa_ok": descriptors.get("tpsa", 0) <= 140
            }
            
            return {
                "lipinski_violations": lipinski_violations,
                "lipinski_compliant": lipinski_violations <= 1,
                "lipinski_rules": lipinski_rules,
                "veber_rules": veber_rules,
                "drug_like_score": max(0, 1 - (lipinski_violations * 0.25))
            }
            
        except Exception as e:
            logger.error(f"Error calculating drug-likeness: {e}")
            return {"lipinski_violations": 5, "lipinski_compliant": False}
    
    def _calculate_fingerprint(self, mol, smiles: str) -> List[int]:
        """Calculate molecular fingerprint for similarity calculations."""
        try:
            if self.rdkit_available:
                from rdkit.Chem import rdMolDescriptors
                # Morgan fingerprint (ECFP4)
                fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
                return list(fp.ToBitString())
            else:
                # Mock fingerprint based on SMILES
                fingerprint = [0] * 1024
                for i, char in enumerate(smiles):
                    if i < 1024:
                        fingerprint[i] = ord(char) % 2
                return fingerprint
                
        except Exception as e:
            logger.error(f"Error calculating fingerprint: {e}")
            return [0] * 1024
    
    def calculate_similarity(self, smiles1: str, smiles2: str) -> float:
        """Calculate Tanimoto similarity between two molecules."""
        try:
            mol1_data = self.process_smiles(smiles1)
            mol2_data = self.process_smiles(smiles2)
            
            if not (mol1_data["valid"] and mol2_data["valid"]):
                return 0.0
            
            fp1 = mol1_data["fingerprint"]
            fp2 = mol2_data["fingerprint"]
            
            # Tanimoto similarity
            intersection = sum(a & b for a, b in zip(fp1, fp2))
            union = sum(a | b for a, b in zip(fp1, fp2))
            
            return intersection / union if union > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating similarity: {e}")
            return 0.0
    
    def batch_process_smiles(self, smiles_list: List[str]) -> List[Dict[str, Any]]:
        """Process multiple SMILES strings in batch."""
        results = []
        for smiles in smiles_list:
            result = self.process_smiles(smiles)
            results.append(result)
        
        logger.info(f"Processed {len(results)} SMILES strings")
        return results
    
    def validate_smiles(self, smiles: str) -> bool:
        """Validate SMILES string."""
        try:
            mol = self.Chem.MolFromSmiles(smiles)
            return mol is not None
        except:
            return False