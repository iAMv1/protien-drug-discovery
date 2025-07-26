"""
Enhanced Data Preprocessing for Protein-Drug Discovery
Comprehensive preprocessing pipeline with detailed analysis
"""

import pandas as pd
import numpy as np
import re
from typing import Dict, List, Tuple, Optional, Any
import logging
from pathlib import Path
from collections import Counter
import json

logging.basicConfig(level=logging.INFO)

class ProteinAnalyzer:
    """Comprehensive protein sequence analysis and preprocessing"""
    
    def __init__(self):
        # Standard amino acid properties
        self.amino_acid_properties = {
            'A': {'name': 'Alanine', 'type': 'Nonpolar', 'charge': 'Neutral', 'hydrophobic': True, 'mw': 89.1},
            'R': {'name': 'Arginine', 'type': 'Basic', 'charge': 'Positive', 'hydrophobic': False, 'mw': 174.2},
            'N': {'name': 'Asparagine', 'type': 'Polar', 'charge': 'Neutral', 'hydrophobic': False, 'mw': 132.1},
            'D': {'name': 'Aspartic acid', 'type': 'Acidic', 'charge': 'Negative', 'hydrophobic': False, 'mw': 133.1},
            'C': {'name': 'Cysteine', 'type': 'Polar', 'charge': 'Neutral', 'hydrophobic': False, 'mw': 121.2},
            'Q': {'name': 'Glutamine', 'type': 'Polar', 'charge': 'Neutral', 'hydrophobic': False, 'mw': 146.1},
            'E': {'name': 'Glutamic acid', 'type': 'Acidic', 'charge': 'Negative', 'hydrophobic': False, 'mw': 147.1},
            'G': {'name': 'Glycine', 'type': 'Nonpolar', 'charge': 'Neutral', 'hydrophobic': False, 'mw': 75.1},
            'H': {'name': 'Histidine', 'type': 'Basic', 'charge': 'Positive', 'hydrophobic': False, 'mw': 155.2},
            'I': {'name': 'Isoleucine', 'type': 'Nonpolar', 'charge': 'Neutral', 'hydrophobic': True, 'mw': 131.2},
            'L': {'name': 'Leucine', 'type': 'Nonpolar', 'charge': 'Neutral', 'hydrophobic': True, 'mw': 131.2},
            'K': {'name': 'Lysine', 'type': 'Basic', 'charge': 'Positive', 'hydrophobic': False, 'mw': 146.2},
            'M': {'name': 'Methionine', 'type': 'Nonpolar', 'charge': 'Neutral', 'hydrophobic': True, 'mw': 149.2},
            'F': {'name': 'Phenylalanine', 'type': 'Aromatic', 'charge': 'Neutral', 'hydrophobic': True, 'mw': 165.2},
            'P': {'name': 'Proline', 'type': 'Nonpolar', 'charge': 'Neutral', 'hydrophobic': False, 'mw': 115.1},
            'S': {'name': 'Serine', 'type': 'Polar', 'charge': 'Neutral', 'hydrophobic': False, 'mw': 105.1},
            'T': {'name': 'Threonine', 'type': 'Polar', 'charge': 'Neutral', 'hydrophobic': False, 'mw': 119.1},
            'W': {'name': 'Tryptophan', 'type': 'Aromatic', 'charge': 'Neutral', 'hydrophobic': True, 'mw': 204.2},
            'Y': {'name': 'Tyrosine', 'type': 'Aromatic', 'charge': 'Neutral', 'hydrophobic': False, 'mw': 181.2},
            'V': {'name': 'Valine', 'type': 'Nonpolar', 'charge': 'Neutral', 'hydrophobic': True, 'mw': 117.1}
        }
    
    def get_amino_acid_info(self, aa: str) -> Dict[str, Any]:
        """Get detailed information about an amino acid"""
        return self.amino_acid_properties.get(aa.upper(), {
            'name': 'Unknown',
            'type': 'Unknown',
            'charge': 'Unknown',
            'hydrophobic': False,
            'mw': 0
        })
    
    def validate_sequence(self, sequence: str) -> Dict[str, Any]:
        """Validate and analyze protein sequence"""
        sequence = sequence.upper().replace(' ', '').replace('\n', '')
        
        # Basic validation
        valid_aa = set(self.amino_acid_properties.keys())
        invalid_chars = set(sequence) - valid_aa
        
        result = {
            'is_valid': len(invalid_chars) == 0,
            'sequence': sequence,
            'length': len(sequence),
            'invalid_characters': list(invalid_chars),
            'composition': self._analyze_composition(sequence),
            'properties': self._analyze_properties(sequence)
        }
        
        return result
    
    def _analyze_composition(self, sequence: str) -> Dict[str, Any]:
        """Analyze amino acid composition"""
        composition = Counter(sequence)
        total = len(sequence)
        
        # Calculate percentages
        composition_pct = {aa: (count/total)*100 for aa, count in composition.items()}
        
        # Group by properties
        property_groups = {
            'hydrophobic': 0,
            'polar': 0,
            'charged': 0,
            'aromatic': 0,
            'basic': 0,
            'acidic': 0
        }
        
        for aa in sequence:
            if aa in self.amino_acid_properties:
                props = self.amino_acid_properties[aa]
                if props['hydrophobic']:
                    property_groups['hydrophobic'] += 1
                if props['type'] in ['Polar', 'Basic', 'Acidic']:
                    property_groups['polar'] += 1
                if props['charge'] in ['Positive', 'Negative']:
                    property_groups['charged'] += 1
                if props['type'] == 'Aromatic':
                    property_groups['aromatic'] += 1
                if props['type'] == 'Basic':
                    property_groups['basic'] += 1
                if props['type'] == 'Acidic':
                    property_groups['acidic'] += 1
        
        # Convert to percentages
        property_groups_pct = {k: (v/total)*100 for k, v in property_groups.items()}
        
        return {
            'amino_acid_counts': dict(composition),
            'amino_acid_percentages': composition_pct,
            'property_groups': property_groups,
            'property_percentages': property_groups_pct
        }
    
    def _analyze_properties(self, sequence: str) -> Dict[str, Any]:
        """Analyze protein properties"""
        # Calculate molecular weight
        molecular_weight = sum(self.amino_acid_properties.get(aa, {'mw': 0})['mw'] for aa in sequence)
        
        # Calculate isoelectric point (simplified)
        basic_count = sum(1 for aa in sequence if self.amino_acid_properties.get(aa, {}).get('charge') == 'Positive')
        acidic_count = sum(1 for aa in sequence if self.amino_acid_properties.get(aa, {}).get('charge') == 'Negative')
        
        # Simplified pI calculation
        if basic_count > acidic_count:
            estimated_pi = 8.5 + (basic_count - acidic_count) * 0.1
        elif acidic_count > basic_count:
            estimated_pi = 5.5 - (acidic_count - basic_count) * 0.1
        else:
            estimated_pi = 7.0
        
        return {
            'molecular_weight': molecular_weight,
            'estimated_pi': estimated_pi,
            'basic_residues': basic_count,
            'acidic_residues': acidic_count
        }

class DrugAnalyzer:
    """Comprehensive drug molecule analysis and preprocessing"""
    
    def __init__(self):
        # Functional groups for SMILES analysis
        self.functional_groups = {
            'hydroxyl': r'O[H]',
            'carbonyl': r'C=O',
            'carboxyl': r'C\(=O\)O',
            'amino': r'N\([^=]',
            'benzene': r'c1ccccc1',
            'pyridine': r'c1ccncc1'
        }
        
        # Atomic weights for molecular weight calculation
        self.atomic_weights = {
            'C': 12.01, 'N': 14.01, 'O': 16.00, 'S': 32.07, 
            'P': 30.97, 'F': 19.00, 'Cl': 35.45, 'Br': 79.90, 'I': 126.90
        }
    
    def analyze_smiles(self, smiles: str) -> Dict[str, Any]:
        """Comprehensive SMILES analysis with RDKit integration when available"""
        smiles = smiles.strip()
        
        result = {
            'smiles': smiles,
            'is_valid': self._validate_smiles(smiles),
            'basic_properties': {},
            'atomic_composition': {},
            'functional_groups': {},
            'drug_likeness': {},
            'complexity': {},
            'molecular_descriptors': {},
            'rdkit_properties': {},
            'errors': []
        }
        
        # Try RDKit analysis first
        rdkit_success = self._analyze_with_rdkit(smiles, result)
        
        if not rdkit_success:
            # Fallback to basic analysis
            result['basic_properties'] = self._calculate_basic_properties(smiles)
            result['atomic_composition'] = self._get_atomic_composition(smiles)
            result['functional_groups'] = self._identify_functional_groups(smiles)
            result['drug_likeness'] = self._assess_drug_likeness(smiles)
            result['complexity'] = self._calculate_complexity(smiles)
            result['molecular_descriptors'] = self._calculate_descriptors(smiles)
        
        return result
    
    def _analyze_with_rdkit(self, smiles: str, result: Dict[str, Any]) -> bool:
        """Analyze SMILES using RDKit if available"""
        try:
            from rdkit import Chem
            from rdkit.Chem import Descriptors, rdMolDescriptors, Crippen, Lipinski
            
            # Create molecule object
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                result['errors'].append("Invalid SMILES - RDKit could not parse")
                result['is_valid'] = False
                return False
            
            result['is_valid'] = True
            
            # RDKit-based properties
            result['rdkit_properties'] = {
                'molecular_weight': Descriptors.MolWt(mol),
                'logp': Crippen.MolLogP(mol),
                'tpsa': Descriptors.TPSA(mol),
                'hbd': rdMolDescriptors.CalcNumHBD(mol),
                'hba': rdMolDescriptors.CalcNumHBA(mol),
                'rotatable_bonds': rdMolDescriptors.CalcNumRotatableBonds(mol),
                'aromatic_rings': rdMolDescriptors.CalcNumAromaticRings(mol),
                'aliphatic_rings': rdMolDescriptors.CalcNumAliphaticRings(mol),
                'heteroatoms': rdMolDescriptors.CalcNumHeteroatoms(mol),
                'formal_charge': Chem.rdmolops.GetFormalCharge(mol),
                'num_atoms': mol.GetNumAtoms(),
                'num_bonds': mol.GetNumBonds(),
                'molecular_formula': rdMolDescriptors.CalcMolFormula(mol)
            }
            
            # Atomic composition from RDKit
            atomic_composition = {}
            for atom in mol.GetAtoms():
                symbol = atom.GetSymbol()
                atomic_composition[symbol] = atomic_composition.get(symbol, 0) + 1
            result['atomic_composition'] = atomic_composition
            
            # Basic properties using RDKit values
            result['basic_properties'] = {
                'estimated_molecular_weight': result['rdkit_properties']['molecular_weight'],
                'estimated_logp': result['rdkit_properties']['logp'],
                'estimated_tpsa': result['rdkit_properties']['tpsa'],
                'rotatable_bonds': result['rdkit_properties']['rotatable_bonds'],
                'formal_charge': result['rdkit_properties']['formal_charge']
            }
            
            # Enhanced drug-likeness assessment
            result['drug_likeness'] = {
                'lipinski_violations': self._calculate_lipinski_violations_rdkit(result['rdkit_properties']),
                'hbd_estimate': result['rdkit_properties']['hbd'],
                'hba_estimate': result['rdkit_properties']['hba'],
                'passes_lipinski': self._passes_lipinski_rdkit(result['rdkit_properties']),
                'qed_score': self._calculate_qed_score(mol) if mol else 0.0
            }
            
            # Enhanced complexity analysis
            result['complexity'] = {
                'structural_complexity': self._calculate_structural_complexity_rdkit(mol),
                'synthetic_accessibility': self._estimate_synthetic_accessibility_rdkit(mol),
                'ring_systems': result['rdkit_properties']['aromatic_rings'] + result['rdkit_properties']['aliphatic_rings'],
                'bertz_complexity': self._calculate_bertz_complexity(mol)
            }
            
            # Functional groups analysis
            result['functional_groups'] = self._identify_functional_groups_rdkit(mol)
            
            # Molecular descriptors
            result['molecular_descriptors'] = self._calculate_descriptors_rdkit(mol)
            
            return True
            
        except ImportError:
            logging.info("RDKit not available, using fallback analysis")
            return False
        except Exception as e:
            result['errors'].append(f"RDKit analysis error: {str(e)}")
            return False
    
    def _validate_smiles(self, smiles: str) -> bool:
        """Validate SMILES string"""
        if not smiles or len(smiles) < 3:
            return False
        
        # Check for balanced parentheses and brackets
        paren_count = smiles.count('(') - smiles.count(')')
        bracket_count = smiles.count('[') - smiles.count(']')
        
        return paren_count == 0 and bracket_count == 0
    
    def _calculate_basic_properties(self, smiles: str) -> Dict[str, Any]:
        """Calculate basic molecular properties from SMILES"""
        # Count atoms (simplified)
        carbon_count = smiles.count('C') + smiles.count('c')
        nitrogen_count = smiles.count('N') + smiles.count('n')
        oxygen_count = smiles.count('O') + smiles.count('o')
        sulfur_count = smiles.count('S') + smiles.count('s')
        
        # Estimate molecular weight
        estimated_mw = (carbon_count * self.atomic_weights['C'] + 
                       nitrogen_count * self.atomic_weights['N'] + 
                       oxygen_count * self.atomic_weights['O'] + 
                       sulfur_count * self.atomic_weights['S'] + 
                       len(smiles) * 0.5 * 1.01)  # Rough H estimate
        
        # Estimate LogP (lipophilicity)
        estimated_logp = (carbon_count * 0.2 - nitrogen_count * 0.7 - 
                         oxygen_count * 0.4 + 1.5)
        
        # Estimate TPSA (Topological Polar Surface Area)
        estimated_tpsa = nitrogen_count * 12 + oxygen_count * 20 + sulfur_count * 25
        
        return {
            'atom_counts': {
                'carbon': carbon_count,
                'nitrogen': nitrogen_count,
                'oxygen': oxygen_count,
                'sulfur': sulfur_count
            },
            'estimated_molecular_weight': estimated_mw,
            'estimated_logp': estimated_logp,
            'estimated_tpsa': estimated_tpsa,
            'heavy_atom_count': carbon_count + nitrogen_count + oxygen_count + sulfur_count
        }
    
    def _identify_functional_groups(self, smiles: str) -> Dict[str, Any]:
        """Identify functional groups in SMILES"""
        found_groups = {}
        
        for group_name, pattern in self.functional_groups.items():
            matches = re.findall(pattern, smiles, re.IGNORECASE)
            found_groups[group_name] = len(matches)
        
        return {
            'functional_groups': found_groups,
            'total_functional_groups': sum(found_groups.values()),
            'diversity_score': len([g for g, count in found_groups.items() if count > 0])
        }
    
    def _assess_drug_likeness(self, smiles: str) -> Dict[str, Any]:
        """Assess drug-likeness using Lipinski's Rule of Five"""
        props = self._calculate_basic_properties(smiles)
        
        # Lipinski's Rule of Five
        mw = props['estimated_molecular_weight']
        logp = props['estimated_logp']
        
        # Estimate H-bond donors and acceptors
        hbd_estimate = smiles.count('OH') + smiles.count('NH')
        hba_estimate = props['atom_counts']['nitrogen'] + props['atom_counts']['oxygen']
        
        violations = 0
        rules = {}
        
        # Rule 1: MW <= 500
        rules['molecular_weight'] = mw <= 500
        if not rules['molecular_weight']:
            violations += 1
        
        # Rule 2: LogP <= 5
        rules['logp'] = logp <= 5
        if not rules['logp']:
            violations += 1
        
        # Rule 3: HBD <= 5
        rules['hbd'] = hbd_estimate <= 5
        if not rules['hbd']:
            violations += 1
        
        # Rule 4: HBA <= 10
        rules['hba'] = hba_estimate <= 10
        if not rules['hba']:
            violations += 1
        
        drug_likeness = "Excellent" if violations == 0 else \
                       "Good" if violations == 1 else \
                       "Poor" if violations == 2 else "Very Poor"
        
        return {
            'lipinski_violations': violations,
            'lipinski_rules': rules,
            'drug_likeness_score': drug_likeness,
            'estimated_values': {
                'molecular_weight': mw,
                'logp': logp,
                'hbd': hbd_estimate,
                'hba': hba_estimate
            }
        }
    
    def _calculate_descriptors(self, smiles: str) -> Dict[str, Any]:
        """Calculate additional molecular descriptors"""
        # Ring analysis
        ring_count = smiles.count('1') + smiles.count('2') + smiles.count('3')
        aromatic_rings = smiles.count('c')
        
        # Rotatable bonds (simplified)
        rotatable_bonds = smiles.count('-') + smiles.count('=') - ring_count
        
        return {
            'ring_count': ring_count,
            'aromatic_rings': aromatic_rings,
            'rotatable_bonds': max(0, rotatable_bonds),
            'smiles_length': len(smiles),
            'character_diversity': len(set(smiles))
        }    

    def _get_atomic_composition(self, smiles: str) -> Dict[str, int]:
        """Get atomic composition from SMILES (fallback method)"""
        composition = {}
        
        # Count atoms (simplified)
        composition['C'] = smiles.count('C') + smiles.count('c')
        composition['N'] = smiles.count('N') + smiles.count('n')
        composition['O'] = smiles.count('O') + smiles.count('o')
        composition['S'] = smiles.count('S') + smiles.count('s')
        composition['P'] = smiles.count('P') + smiles.count('p')
        composition['F'] = smiles.count('F') + smiles.count('f')
        composition['Cl'] = smiles.count('Cl')
        composition['Br'] = smiles.count('Br')
        composition['I'] = smiles.count('I')
        
        # Remove zero counts
        return {k: v for k, v in composition.items() if v > 0}
    
    def _calculate_complexity(self, smiles: str) -> Dict[str, Any]:
        """Calculate molecular complexity (fallback method)"""
        ring_count = smiles.count('1') + smiles.count('2') + smiles.count('3')
        branch_count = smiles.count('(')
        
        # Simple complexity score
        complexity_score = (len(smiles) / 20.0) + (ring_count * 0.5) + (branch_count * 0.3)
        
        # Simple synthetic accessibility estimate
        sa_score = min(10.0, 1.0 + complexity_score)
        
        return {
            'structural_complexity': min(10.0, complexity_score),
            'synthetic_accessibility': sa_score,
            'ring_systems': ring_count,
            'branching_points': branch_count
        }
    
    # RDKit-specific helper methods
    def _calculate_lipinski_violations_rdkit(self, props: Dict[str, float]) -> int:
        """Calculate Lipinski violations using RDKit properties"""
        violations = 0
        
        if props['molecular_weight'] > 500:
            violations += 1
        if props['logp'] > 5:
            violations += 1
        if props['hbd'] > 5:
            violations += 1
        if props['hba'] > 10:
            violations += 1
        
        return violations
    
    def _passes_lipinski_rdkit(self, props: Dict[str, float]) -> bool:
        """Check if molecule passes Lipinski's Rule of Five"""
        return self._calculate_lipinski_violations_rdkit(props) <= 1
    
    def _calculate_qed_score(self, mol) -> float:
        """Calculate QED (Quantitative Estimate of Drug-likeness) score"""
        try:
            from rdkit.Chem import QED
            return QED.qed(mol)
        except:
            return 0.0
    
    def _calculate_structural_complexity_rdkit(self, mol) -> float:
        """Calculate structural complexity using RDKit"""
        try:
            from rdkit.Chem import rdMolDescriptors
            
            # Combine multiple complexity measures
            num_atoms = mol.GetNumAtoms()
            num_bonds = mol.GetNumBonds()
            num_rings = rdMolDescriptors.CalcNumRings(mol)
            num_heteroatoms = rdMolDescriptors.CalcNumHeteroatoms(mol)
            
            # Normalized complexity score
            complexity = (
                (num_atoms / 50.0) * 0.3 +
                (num_bonds / 60.0) * 0.3 +
                (num_rings / 5.0) * 0.2 +
                (num_heteroatoms / 10.0) * 0.2
            ) * 10
            
            return min(10.0, complexity)
        except:
            return 5.0  # Default complexity
    
    def _estimate_synthetic_accessibility_rdkit(self, mol) -> float:
        """Estimate synthetic accessibility using RDKit"""
        try:
            from rdkit.Contrib.SA_Score import sascorer
            return sascorer.calculateScore(mol)
        except:
            # Fallback estimation
            try:
                from rdkit.Chem import rdMolDescriptors
                num_rings = len([ring for ring in mol.GetRingInfo().AtomRings()])
                rotatable_bonds = rdMolDescriptors.CalcNumRotatableBonds(mol)
                
                # Simple heuristic
                sa_score = 1.0 + (num_rings * 0.5) + (rotatable_bonds * 0.1)
                return min(10.0, sa_score)
            except:
                return 5.0  # Default SA score
    
    def _calculate_bertz_complexity(self, mol) -> float:
        """Calculate Bertz complexity index"""
        try:
            from rdkit.Chem import rdMolDescriptors
            return rdMolDescriptors.BertzCT(mol)
        except:
            return 0.0
    
    def _identify_functional_groups_rdkit(self, mol) -> Dict[str, Any]:
        """Identify functional groups using RDKit"""
        try:
            from rdkit import Chem
            
            # Define functional group patterns
            functional_groups = {
                'hydroxyl': Chem.MolFromSmarts('[OH]'),
                'carbonyl': Chem.MolFromSmarts('[CX3]=[OX1]'),
                'carboxyl': Chem.MolFromSmarts('C(=O)[OH]'),
                'amino': Chem.MolFromSmarts('[NX3;H2,H1;!$(NC=O)]'),
                'benzene': Chem.MolFromSmarts('c1ccccc1'),
                'pyridine': Chem.MolFromSmarts('c1ccncc1'),
                'ether': Chem.MolFromSmarts('[OD2]([#6])[#6]'),
                'ester': Chem.MolFromSmarts('[CX3](=O)[OX2H0][#6]'),
                'amide': Chem.MolFromSmarts('[CX3](=[OX1])[NX3H2]'),
                'nitro': Chem.MolFromSmarts('[NX3+](=O)[O-]'),
                'sulfone': Chem.MolFromSmarts('[SX4](=O)(=O)'),
                'phosphate': Chem.MolFromSmarts('[PX4](=O)([OH])([OH])[OH]')
            }
            
            found_groups = {}
            for group_name, pattern in functional_groups.items():
                if pattern:
                    matches = mol.GetSubstructMatches(pattern)
                    found_groups[group_name] = len(matches)
                else:
                    found_groups[group_name] = 0
            
            return {
                'functional_groups': found_groups,
                'total_functional_groups': sum(found_groups.values()),
                'diversity_score': len([g for g, count in found_groups.items() if count > 0])
            }
            
        except Exception as e:
            # Fallback to basic analysis
            return self._identify_functional_groups(self.smiles if hasattr(self, 'smiles') else '')
    
    def _calculate_descriptors_rdkit(self, mol) -> Dict[str, Any]:
        """Calculate molecular descriptors using RDKit"""
        try:
            from rdkit.Chem import rdMolDescriptors, Descriptors
            
            descriptors = {
                'ring_count': rdMolDescriptors.CalcNumRings(mol),
                'aromatic_rings': rdMolDescriptors.CalcNumAromaticRings(mol),
                'aliphatic_rings': rdMolDescriptors.CalcNumAliphaticRings(mol),
                'rotatable_bonds': rdMolDescriptors.CalcNumRotatableBonds(mol),
                'heavy_atoms': mol.GetNumHeavyAtoms(),
                'formal_charge': Chem.rdmolops.GetFormalCharge(mol),
                'molar_refractivity': Descriptors.MolMR(mol),
                'balaban_j': Descriptors.BalabanJ(mol),
                'kappa1': Descriptors.Kappa1(mol),
                'kappa2': Descriptors.Kappa2(mol),
                'kappa3': Descriptors.Kappa3(mol),
                'chi0v': Descriptors.Chi0v(mol),
                'chi1v': Descriptors.Chi1v(mol),
                'chi2v': Descriptors.Chi2v(mol),
                'hall_kier_alpha': Descriptors.HallKierAlpha(mol),
                'slogp_vsa': [
                    Descriptors.SlogP_VSA1(mol), Descriptors.SlogP_VSA2(mol),
                    Descriptors.SlogP_VSA3(mol), Descriptors.SlogP_VSA4(mol)
                ],
                'smr_vsa': [
                    Descriptors.SMR_VSA1(mol), Descriptors.SMR_VSA2(mol),
                    Descriptors.SMR_VSA3(mol), Descriptors.SMR_VSA4(mol)
                ]
            }
            
            return descriptors
            
        except Exception as e:
            # Fallback to basic descriptors
            return {
                'ring_count': 0,
                'aromatic_rings': 0,
                'rotatable_bonds': 0,
                'heavy_atoms': mol.GetNumHeavyAtoms() if mol else 0,
                'error': f"RDKit descriptor calculation failed: {str(e)}"
            }


class EnhancedPreprocessor:
    """Enhanced preprocessing pipeline for protein-drug discovery"""
    
    def __init__(self):
        self.protein_analyzer = ProteinAnalyzer()
        self.drug_analyzer = DrugAnalyzer()
    
    def preprocess_dataset(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Comprehensive dataset preprocessing"""
        logging.info("Starting enhanced preprocessing...")
        
        results = {
            'original_size': len(df),
            'protein_analysis': {},
            'drug_analysis': {},
            'interaction_analysis': {},
            'quality_metrics': {},
            'processed_data': None
        }
        
        # Analyze proteins
        protein_results = []
        for protein_seq in df['protein'].unique():
            analysis = self.protein_analyzer.validate_sequence(protein_seq)
            protein_results.append(analysis)
        
        results['protein_analysis'] = self._summarize_protein_analysis(protein_results)
        
        # Analyze drugs
        drug_results = []
        for drug_smiles in df['compound'].unique():
            analysis = self.drug_analyzer.analyze_smiles(drug_smiles)
            drug_results.append(analysis)
        
        results['drug_analysis'] = self._summarize_drug_analysis(drug_results)
        
        # Analyze interactions
        results['interaction_analysis'] = self._analyze_interactions(df)
        
        # Quality assessment
        results['quality_metrics'] = self._assess_data_quality(df, protein_results, drug_results)
        
        # Clean and process data
        results['processed_data'] = self._clean_data(df, protein_results, drug_results)
        
        logging.info("Enhanced preprocessing completed")
        return results
    
    def analyze_single_protein(self, protein_seq: str) -> Dict[str, Any]:
        """Analyze a single protein sequence"""
        return self.protein_analyzer.validate_sequence(protein_seq)
    
    def analyze_single_drug(self, smiles: str) -> Dict[str, Any]:
        """Analyze a single drug SMILES"""
        return self.drug_analyzer.analyze_smiles(smiles)
    
    def _summarize_protein_analysis(self, protein_results: List[Dict]) -> Dict[str, Any]:
        """Summarize protein analysis results"""
        valid_proteins = [p for p in protein_results if p['is_valid']]
        
        if not valid_proteins:
            return {'error': 'No valid proteins found'}
        
        # Length statistics
        lengths = [p['length'] for p in valid_proteins]
        
        # Property statistics
        hydrophobic_pcts = [p['composition']['property_percentages']['hydrophobic'] for p in valid_proteins]
        
        return {
            'total_proteins': len(protein_results),
            'valid_proteins': len(valid_proteins),
            'length_stats': {
                'min': min(lengths),
                'max': max(lengths),
                'mean': np.mean(lengths),
                'median': np.median(lengths)
            },
            'property_stats': {
                'avg_hydrophobic_pct': np.mean(hydrophobic_pcts),
                'avg_molecular_weight': np.mean([p['properties']['molecular_weight'] for p in valid_proteins])
            }
        }
    
    def _summarize_drug_analysis(self, drug_results: List[Dict]) -> Dict[str, Any]:
        """Summarize drug analysis results"""
        valid_drugs = [d for d in drug_results if d['is_valid']]
        
        if not valid_drugs:
            return {'error': 'No valid drugs found'}
        
        # Drug-likeness statistics
        drug_likeness_scores = [d['drug_likeness']['lipinski_violations'] for d in valid_drugs]
        
        return {
            'total_drugs': len(drug_results),
            'valid_drugs': len(valid_drugs),
            'drug_likeness_stats': {
                'avg_violations': np.mean(drug_likeness_scores),
                'drug_like_compounds': sum(1 for score in drug_likeness_scores if score <= 1),
                'drug_like_percentage': (sum(1 for score in drug_likeness_scores if score <= 1) / len(valid_drugs)) * 100
            },
            'molecular_stats': {
                'avg_molecular_weight': np.mean([d['basic_properties']['estimated_molecular_weight'] for d in valid_drugs]),
                'avg_logp': np.mean([d['basic_properties']['estimated_logp'] for d in valid_drugs])
            }
        }
    
    def _analyze_interactions(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze protein-drug interactions"""
        return {
            'total_interactions': len(df),
            'unique_proteins': df['protein'].nunique(),
            'unique_compounds': df['compound'].nunique(),
            'binding_distribution': {
                'positive_interactions': (df['binding_label'] == 1).sum() if 'binding_label' in df.columns else 'N/A',
                'binding_ratio': (df['binding_label'] == 1).mean() if 'binding_label' in df.columns else 'N/A'
            },
            'affinity_stats': {
                'min_pkd': df['pKd'].min() if 'pKd' in df.columns else 'N/A',
                'max_pkd': df['pKd'].max() if 'pKd' in df.columns else 'N/A',
                'mean_pkd': df['pKd'].mean() if 'pKd' in df.columns else 'N/A'
            }
        }
    
    def _assess_data_quality(self, df: pd.DataFrame, protein_results: List[Dict], drug_results: List[Dict]) -> Dict[str, Any]:
        """Assess overall data quality"""
        valid_proteins = sum(1 for p in protein_results if p['is_valid'])
        valid_drugs = sum(1 for d in drug_results if d['is_valid'])
        
        return {
            'protein_quality': (valid_proteins / len(protein_results)) * 100 if protein_results else 0,
            'drug_quality': (valid_drugs / len(drug_results)) * 100 if drug_results else 0,
            'missing_values': df.isnull().sum().to_dict(),
            'duplicate_interactions': df.duplicated().sum(),
            'overall_quality_score': ((valid_proteins / len(protein_results) if protein_results else 0) + 
                                    (valid_drugs / len(drug_results) if drug_results else 0)) * 50
        }
    
    def _clean_data(self, df: pd.DataFrame, protein_results: List[Dict], drug_results: List[Dict]) -> pd.DataFrame:
        """Clean and filter the dataset"""
        # Create mapping of valid sequences
        valid_proteins = {p['sequence'] for p in protein_results if p['is_valid']}
        valid_drugs = {d['smiles'] for d in drug_results if d['is_valid']}
        
        # Filter dataset
        cleaned_df = df[
            df['protein'].isin(valid_proteins) & 
            df['compound'].isin(valid_drugs)
        ].copy()
        
        # Remove duplicates
        cleaned_df = cleaned_df.drop_duplicates()
        
        # Add quality flags
        cleaned_df['protein_length'] = cleaned_df['protein'].str.len()
        cleaned_df['compound_length'] = cleaned_df['compound'].str.len()
        
        return cleaned_df