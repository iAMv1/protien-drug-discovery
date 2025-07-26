"""Protein data processing from UniProt and other sources."""

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


class ProteinDataProcessor:
    """Process protein data from UniProt and other databases."""
    
    def __init__(self, cache_dir: str = "data/cache"):
        """Initialize protein data processor."""
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.uniprot_base_url = "https://rest.uniprot.org/uniprotkb"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'ProteinDrugDiscovery/1.0 (research@example.com)'
        })
    
    def fetch_protein_by_id(self, uniprot_id: str) -> Dict[str, Any]:
        """
        Fetch protein data from UniProt by ID.
        
        Args:
            uniprot_id: UniProt accession ID
            
        Returns:
            Dictionary with protein information
        """
        try:
            # Check cache first
            cache_file = self.cache_dir / f"protein_{uniprot_id}.json"
            if cache_file.exists():
                with open(cache_file, 'r') as f:
                    return json.load(f)
            
            # Fetch from UniProt
            url = f"{self.uniprot_base_url}/{uniprot_id}"
            params = {"format": "json"}
            
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            # Extract relevant information
            protein_info = self._extract_protein_info(data)
            
            # Cache the result
            with open(cache_file, 'w') as f:
                json.dump(protein_info, f, indent=2)
            
            logger.info(f"Fetched protein data for {uniprot_id}")
            return protein_info
            
        except requests.RequestException as e:
            logger.error(f"Error fetching protein {uniprot_id}: {e}")
            return self._create_mock_protein_data(uniprot_id)
        except Exception as e:
            logger.error(f"Unexpected error for protein {uniprot_id}: {e}")
            return self._create_mock_protein_data(uniprot_id)
    
    def _extract_protein_info(self, uniprot_data: Dict) -> Dict[str, Any]:
        """Extract relevant information from UniProt JSON response."""
        try:
            # Basic information
            protein_info = {
                "uniprot_id": uniprot_data.get("primaryAccession", ""),
                "protein_name": "",
                "gene_name": "",
                "organism": "",
                "sequence": "",
                "length": 0,
                "function": "",
                "subcellular_location": [],
                "domains": [],
                "keywords": []
            }
            
            # Protein names
            if "proteinDescription" in uniprot_data:
                desc = uniprot_data["proteinDescription"]
                if "recommendedName" in desc:
                    protein_info["protein_name"] = desc["recommendedName"].get("fullName", {}).get("value", "")
            
            # Gene name
            if "genes" in uniprot_data and uniprot_data["genes"]:
                gene = uniprot_data["genes"][0]
                if "geneName" in gene:
                    protein_info["gene_name"] = gene["geneName"].get("value", "")
            
            # Organism
            if "organism" in uniprot_data:
                organism = uniprot_data["organism"]
                if "scientificName" in organism:
                    protein_info["organism"] = organism["scientificName"]
            
            # Sequence
            if "sequence" in uniprot_data:
                seq_data = uniprot_data["sequence"]
                protein_info["sequence"] = seq_data.get("value", "")
                protein_info["length"] = seq_data.get("length", 0)
            
            # Function
            if "comments" in uniprot_data:
                for comment in uniprot_data["comments"]:
                    if comment.get("commentType") == "FUNCTION":
                        if "texts" in comment and comment["texts"]:
                            protein_info["function"] = comment["texts"][0].get("value", "")
                        break
            
            # Keywords
            if "keywords" in uniprot_data:
                protein_info["keywords"] = [kw.get("value", "") for kw in uniprot_data["keywords"]]
            
            return protein_info
            
        except Exception as e:
            logger.error(f"Error extracting protein info: {e}")
            return self._create_mock_protein_data("unknown")
    
    def _create_mock_protein_data(self, uniprot_id: str) -> Dict[str, Any]:
        """Create mock protein data for testing."""
        mock_sequences = {
            "P04637": "MEEPQSDPSVEPPLSQETFSDLWKLLPENNVLSPLPSQAMDDLMLSPDDIEQWFTEDPGPDEAPRMPEAAPPVAPAPAAPTPAAPAPAPSWPLSSSVPSQKTYQGSYGFRLGFLHSGTAKSVTCTYSPALNKMFCQLAKTCPVQLWVDSTPPPGTRVRAMAIYKQSQHMTEVVRRCPHHERCSDSDGLAPPQHLIRVEGNLRVEYLDDRNTFRHSVVVPYEPPEVGSDCTTIHYNYMCNSSCMGGMNRRPILTIITLEDSSGNLLGRNSFEVRVCACPGRDRRTEEENLRKKGEPHHELPPGSTKRALPNNTSSSPQPKKKPLDGEYFTLQIRGRERFEMFRELNEALELKDAQAGKEPGGSRAHSSHLKSKKGQSTSRHKKLMFKTEGPDSD",
            "P53_HUMAN": "MEEPQSDPSVEPPLSQETFSDLWKLLPENNVLSPLPSQAMDDLMLSPDDIEQWFTEDPGPDEAPRMPEAAPPVAPAPAAPTPAAPAPAPSWPLSSSVPSQKTYQGSYGFRLGFLHSGTAKSVTCTYSPALNKMFCQLAKTCPVQLWVDSTPPPGTRVRAMAIYKQSQHMTEVVRRCPHHERCSDSDGLAPPQHLIRVEGNLRVEYLDDRNTFRHSVVVPYEPPEVGSDCTTIHYNYMCNSSCMGGMNRRPILTIITLEDSSGNLLGRNSFEVRVCACPGRDRRTEEENLRKKGEPHHELPPGSTKRALPNNTSSSPQPKKKPLDGEYFTLQIRGRERFEMFRELNEALELKDAQAGKEPGGSRAHSSHLKSKKGQSTSRHKKLMFKTEGPDSD"
        }
        
        sequence = mock_sequences.get(uniprot_id, "MKWVTFISLLLLFSSAYSRGVFRRDTHKSEIAHRFKDLGEEHFKGLVLIAFSQYLQQCPFDEHVKLVNELTEFAKTCVADESHAGCEKSLHTLFGDELCKVASLRETYGDMADCCEKQEPERNECFLSHKDDSPDLPKLKPDPNTLCDEFKADEKKFWGKYLYEIARRHPYFYAPELLYYANKYNGVFQECCQAEDKGACLLPKIETMREKVLASSARQRLRCASIQKFGERALKAWSVARLSQKFPKAEFVEVTKLVTDLTKVHKECCHGDLLECADDRADLAKYICDNQDTISSKLKECCDKPVNGFNLYNKQRQVPQSRGNQVQGJINVTDSWSKINVFLGQVS")
        
        return {
            "uniprot_id": uniprot_id,
            "protein_name": f"Mock protein {uniprot_id}",
            "gene_name": f"GENE_{uniprot_id}",
            "organism": "Homo sapiens",
            "sequence": sequence,
            "length": len(sequence),
            "function": f"Mock function for {uniprot_id}",
            "subcellular_location": ["Cytoplasm"],
            "domains": ["Mock domain"],
            "keywords": ["Mock", "Protein"],
            "mock": True
        }
    
    def search_proteins_by_keyword(self, keyword: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search proteins by keyword.
        
        Args:
            keyword: Search keyword
            limit: Maximum number of results
            
        Returns:
            List of protein information dictionaries
        """
        try:
            # Check cache
            cache_file = self.cache_dir / f"search_{keyword}_{limit}.json"
            if cache_file.exists():
                with open(cache_file, 'r') as f:
                    return json.load(f)
            
            # Search UniProt
            url = f"{self.uniprot_base_url}/search"
            params = {
                "query": keyword,
                "format": "json",
                "size": limit
            }
            
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            results = []
            
            for entry in data.get("results", []):
                protein_info = self._extract_protein_info(entry)
                results.append(protein_info)
            
            # Cache results
            with open(cache_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            logger.info(f"Found {len(results)} proteins for keyword '{keyword}'")
            return results
            
        except Exception as e:
            logger.error(f"Error searching proteins with keyword '{keyword}': {e}")
            return self._create_mock_search_results(keyword, limit)
    
    def _create_mock_search_results(self, keyword: str, limit: int) -> List[Dict[str, Any]]:
        """Create mock search results for testing."""
        results = []
        for i in range(min(limit, 5)):  # Create up to 5 mock results
            mock_id = f"MOCK_{keyword.upper()}_{i+1}"
            results.append(self._create_mock_protein_data(mock_id))
        return results
    
    def get_drug_targets(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get known drug target proteins.
        
        Args:
            limit: Maximum number of targets to retrieve
            
        Returns:
            List of drug target protein information
        """
        try:
            # Search for proteins with drug target annotations
            return self.search_proteins_by_keyword("drug target", limit)
        except Exception as e:
            logger.error(f"Error fetching drug targets: {e}")
            return self._create_mock_drug_targets(limit)
    
    def _create_mock_drug_targets(self, limit: int) -> List[Dict[str, Any]]:
        """Create mock drug target data."""
        known_targets = [
            ("P04637", "Tumor protein p53"),
            ("P35354", "Prostaglandin G/H synthase 2"),
            ("P23219", "Prostaglandin G/H synthase 1"),
            ("P08684", "Cytochrome P450 3A4"),
            ("P11712", "Cytochrome P450 2C9")
        ]
        
        results = []
        for i, (uniprot_id, name) in enumerate(known_targets[:limit]):
            mock_data = self._create_mock_protein_data(uniprot_id)
            mock_data["protein_name"] = name
            mock_data["is_drug_target"] = True
            results.append(mock_data)
        
        return results
    
    def validate_protein_sequence(self, sequence: str) -> bool:
        """Validate protein amino acid sequence."""
        if not sequence or not isinstance(sequence, str):
            return False
        
        # Standard amino acid codes
        valid_aa = set('ACDEFGHIKLMNPQRSTVWY')
        sequence_upper = sequence.upper().replace(' ', '').replace('\n', '')
        
        return all(aa in valid_aa for aa in sequence_upper)
    
    def get_sequence_stats(self, sequence: str) -> Dict[str, Any]:
        """Get statistics for a protein sequence."""
        if not self.validate_protein_sequence(sequence):
            return {"valid": False}
        
        sequence = sequence.upper().replace(' ', '').replace('\n', '')
        
        # Amino acid composition
        aa_counts = {}
        for aa in 'ACDEFGHIKLMNPQRSTVWY':
            aa_counts[aa] = sequence.count(aa)
        
        # Basic statistics
        stats = {
            "valid": True,
            "length": len(sequence),
            "amino_acid_counts": aa_counts,
            "molecular_weight": sum(aa_counts[aa] * self._get_aa_weight(aa) for aa in aa_counts),
            "hydrophobic_ratio": sum(aa_counts[aa] for aa in 'AILMFPWV') / len(sequence),
            "charged_ratio": sum(aa_counts[aa] for aa in 'DEKR') / len(sequence)
        }
        
        return stats
    
    def _get_aa_weight(self, aa: str) -> float:
        """Get molecular weight of amino acid."""
        weights = {
            'A': 89.1, 'C': 121.0, 'D': 133.1, 'E': 147.1, 'F': 165.2,
            'G': 75.1, 'H': 155.2, 'I': 131.2, 'K': 146.2, 'L': 131.2,
            'M': 149.2, 'N': 132.1, 'P': 115.1, 'Q': 146.2, 'R': 174.2,
            'S': 105.1, 'T': 119.1, 'V': 117.1, 'W': 204.2, 'Y': 181.2
        }
        return weights.get(aa, 110.0)  # Average weight as default