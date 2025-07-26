"""
Batch processing service for protein-drug discovery system.
Handles job queue management, progress tracking, and result processing.
"""

import json
import uuid
import asyncio
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
from datetime import datetime

logger = logging.getLogger(__name__)

class JobStatus(Enum):
    """Job status enumeration"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class BatchJob:
    """Batch job data structure"""
    job_id: str
    protein_sequences: List[str]
    drug_smiles: List[str]
    status: JobStatus
    created_at: datetime
    results: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None

class BatchService:
    """Service for handling batch protein-drug predictions"""
    
    def __init__(self):
        self.jobs: Dict[str, BatchJob] = {}
        
    def create_job(self, protein_sequences: List[str], drug_smiles: List[str]) -> str:
        """Create a new batch job"""
        job_id = str(uuid.uuid4())
        job = BatchJob(
            job_id=job_id,
            protein_sequences=protein_sequences,
            drug_smiles=drug_smiles,
            status=JobStatus.PENDING,
            created_at=datetime.now()
        )
        self.jobs[job_id] = job
        return job_id
    
    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get job status and results"""
        if job_id not in self.jobs:
            return None
        
        job = self.jobs[job_id]
        return {
            "job_id": job.job_id,
            "status": job.status.value,
            "created_at": job.created_at.isoformat(),
            "results": job.results,
            "error_message": job.error_message
        }
    
    async def process_job(self, job_id: str, predictor=None) -> bool:
        """Process a batch job asynchronously"""
        if job_id not in self.jobs:
            return False
        
        job = self.jobs[job_id]
        job.status = JobStatus.RUNNING
        
        try:
            results = []
            
            # Check if predictor is the realtime inference engine
            if hasattr(predictor, 'predict_batch'):
                # Use optimized batch processing
                logger.info(f"Using realtime inference engine for batch job {job_id}")
                
                # Create all protein-drug combinations
                protein_list = []
                drug_list = []
                for i, protein_seq in enumerate(job.protein_sequences):
                    for j, drug_smiles in enumerate(job.drug_smiles):
                        protein_list.append(protein_seq)
                        drug_list.append(drug_smiles)
                
                # Process batch
                batch_results = await predictor.predict_batch(protein_list, drug_list, f"batch_{job_id}")
                
                # Convert results
                for k, result in enumerate(batch_results):
                    i = k // len(job.drug_smiles)  # protein index
                    j = k % len(job.drug_smiles)   # drug index
                    
                    results.append({
                        "protein_sequence": job.protein_sequences[i][:50] + "...",  # Truncate for storage
                        "drug_smiles": job.drug_smiles[j],
                        "prediction": {
                            "binding_affinity": result.binding_affinity,
                            "confidence": result.confidence,
                            "protein_id": f"protein_{i}",
                            "drug_id": f"drug_{j}",
                            "processing_time_ms": result.processing_time_ms,
                            "cached": result.cached,
                            "model_version": result.model_version
                        }
                    })
                
            elif hasattr(predictor, 'predict_single'):
                # Use single predictions with realtime inference engine
                logger.info(f"Using realtime inference engine (single predictions) for batch job {job_id}")
                
                for i, protein_seq in enumerate(job.protein_sequences):
                    for j, drug_smiles in enumerate(job.drug_smiles):
                        result = await predictor.predict_single(
                            protein_seq, 
                            drug_smiles, 
                            f"batch_{job_id}_{i}_{j}"
                        )
                        
                        results.append({
                            "protein_sequence": protein_seq[:50] + "...",  # Truncate for storage
                            "drug_smiles": drug_smiles,
                            "prediction": {
                                "binding_affinity": result.binding_affinity,
                                "confidence": result.confidence,
                                "protein_id": f"protein_{i}",
                                "drug_id": f"drug_{j}",
                                "processing_time_ms": result.processing_time_ms,
                                "cached": result.cached,
                                "model_version": result.model_version
                            }
                        })
                        
            elif hasattr(predictor, 'predict_interaction'):
                # Use basic prediction service
                logger.info(f"Using basic prediction service for batch job {job_id}")
                
                for i, protein_seq in enumerate(job.protein_sequences):
                    for j, drug_smiles in enumerate(job.drug_smiles):
                        binding_affinity = predictor.predict_interaction(protein_seq, drug_smiles)
                        
                        results.append({
                            "protein_sequence": protein_seq[:50] + "...",  # Truncate for storage
                            "drug_smiles": drug_smiles,
                            "prediction": {
                                "binding_affinity": binding_affinity,
                                "confidence": 0.85,  # Default confidence
                                "protein_id": f"protein_{i}",
                                "drug_id": f"drug_{j}",
                                "processing_time_ms": 0.0,
                                "cached": False,
                                "model_version": "basic_v1.0"
                            }
                        })
            else:
                # Fallback to mock predictions
                logger.warning(f"Using mock predictions for batch job {job_id}")
                
                for i, protein_seq in enumerate(job.protein_sequences):
                    for j, drug_smiles in enumerate(job.drug_smiles):
                        results.append({
                            "protein_sequence": protein_seq[:50] + "...",  # Truncate for storage
                            "drug_smiles": drug_smiles,
                            "prediction": {
                                "binding_affinity": 0.75,
                                "confidence": 0.85,
                                "protein_id": f"protein_{i}",
                                "drug_id": f"drug_{j}",
                                "processing_time_ms": 0.0,
                                "cached": False,
                                "model_version": "mock_v1.0"
                            }
                        })
            
            # Calculate summary statistics
            binding_affinities = [r["prediction"]["binding_affinity"] for r in results]
            avg_binding_affinity = sum(binding_affinities) / len(binding_affinities) if binding_affinities else 0
            high_affinity_count = sum(1 for affinity in binding_affinities if affinity > 8.0)
            
            job.results = {
                "predictions": results,
                "total_processed": len(results),
                "summary": {
                    "avg_binding_affinity": avg_binding_affinity,
                    "high_affinity_count": high_affinity_count,
                    "max_binding_affinity": max(binding_affinities) if binding_affinities else 0,
                    "min_binding_affinity": min(binding_affinities) if binding_affinities else 0
                }
            }
            job.status = JobStatus.COMPLETED
            logger.info(f"Batch job {job_id} completed successfully with {len(results)} predictions")
            return True
            
        except Exception as e:
            job.status = JobStatus.FAILED
            job.error_message = str(e)
            logger.error(f"Job {job_id} failed: {e}")
            return False
    
    def list_jobs(self) -> List[Dict[str, Any]]:
        """List all jobs with their status"""
        return [
            {
                "job_id": job.job_id,
                "status": job.status.value,
                "created_at": job.created_at.isoformat(),
                "protein_count": len(job.protein_sequences),
                "drug_count": len(job.drug_smiles),
                "has_results": job.results is not None
            }
            for job in self.jobs.values()
        ]
    
    def delete_job(self, job_id: str) -> bool:
        """Delete a job and its results"""
        if job_id in self.jobs:
            del self.jobs[job_id]
            return True
        return False