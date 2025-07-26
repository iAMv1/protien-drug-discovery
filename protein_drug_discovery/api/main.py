"""FastAPI backend for protein-drug discovery platform."""

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
import logging
import sys
import os
import time

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from protein_drug_discovery.core import ESMProteinModel, DrugProcessor
from protein_drug_discovery.data import ProteinDataProcessor, DrugDataProcessor
from protein_drug_discovery.core.interaction_predictor import PredictionService
from protein_drug_discovery.core.batch_service import BatchService
from protein_drug_discovery.core.realtime_inference import get_inference_engine, RealtimeInferenceEngine
from protein_drug_discovery.auth.auth_routes import auth_router, workspace_router
from protein_drug_discovery.auth.auth_dependencies import get_current_active_user, require_role
from protein_drug_discovery.auth.auth_models import UserRole

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Protein-Drug Discovery API",
    description="AI-powered protein-drug interaction prediction API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include authentication and workspace routers
app.include_router(auth_router)
app.include_router(workspace_router)

# Global components (initialized on startup)
esm_model = None
drug_processor = None
protein_data = None
drug_data = None
batch_service = None
realtime_engine = None

# Initialize prediction service globally (fallback)
prediction_service = PredictionService(model_path="models/unsloth_finetuned_model")


# Pydantic models for request/response
class ProteinInput(BaseModel):
    sequence: str = Field(..., description="Protein amino acid sequence")
    uniprot_id: Optional[str] = Field(None, description="UniProt ID (alternative to sequence)")


class DrugInput(BaseModel):
    smiles: str = Field(..., description="SMILES string representation")
    chembl_id: Optional[str] = Field(None, description="ChEMBL ID (alternative to SMILES)")


class InteractionRequest(BaseModel):
    protein: ProteinInput
    drug: DrugInput


class InteractionResponse(BaseModel):
    interaction_probability: float
    binding_strength: str
    confidence: float
    protein_properties: Dict[str, Any]
    drug_properties: Dict[str, Any]
    processing_time: float


class ProteinAnalysisResponse(BaseModel):
    sequence_length: int
    molecular_weight: float
    amino_acid_composition: Dict[str, int]
    hydrophobic_ratio: float
    charged_ratio: float
    embedding_dimension: List[int]
    valid: bool


class DrugAnalysisResponse(BaseModel):
    molecular_weight: float
    logp: float
    hbd: int
    hba: int
    tpsa: float
    rotatable_bonds: int
    lipinski_compliant: bool
    drug_like_score: float
    valid: bool


class SearchResponse(BaseModel):
    results: List[Dict[str, Any]]
    total_count: int
    query: str


class BatchJobRequest(BaseModel):
    protein_sequences: List[str] = Field(..., description="List of protein sequences")
    drug_smiles: List[str] = Field(..., description="List of drug SMILES strings")


class BatchJobResponse(BaseModel):
    job_id: str
    status: str
    message: str


class BatchJobStatus(BaseModel):
    job_id: str
    status: str
    created_at: str
    results: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None


@app.on_event("startup")
async def startup_event():
    """Initialize components on startup."""
    global esm_model, drug_processor, protein_data, drug_data, batch_service, realtime_engine
    
    try:
        logger.info("Initializing protein-drug discovery components...")
        
        esm_model = ESMProteinModel(model_size="150M", device="cpu")
        drug_processor = DrugProcessor()
        protein_data = ProteinDataProcessor()
        drug_data = DrugDataProcessor()
        batch_service = BatchService()
        
        # Initialize realtime inference engine
        logger.info("Initializing realtime inference engine...")
        realtime_engine = get_inference_engine(model_path="models/unsloth_finetuned_model")
        logger.info("Realtime inference engine initialized successfully")
        
        logger.info("All components initialized successfully")
        
    except Exception as e:
        logger.error(f"Error initializing components: {e}")
        raise


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Protein-Drug Discovery API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "predict": "/predict - Predict protein-drug interactions",
            "analyze_protein": "/analyze/protein - Analyze protein sequence",
            "analyze_drug": "/analyze/drug - Analyze drug molecule",
            "search_proteins": "/search/proteins - Search protein database",
            "search_drugs": "/search/drugs - Search drug database",
            "docs": "/docs - API documentation"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "components": {
            "esm_model": esm_model is not None,
            "drug_processor": drug_processor is not None,
            "protein_data": protein_data is not None,
            "drug_data": drug_data is not None,
            "realtime_engine": realtime_engine is not None
        }
    }


@app.get("/performance/stats")
async def get_performance_stats():
    """Get comprehensive performance statistics."""
    if not realtime_engine:
        raise HTTPException(status_code=503, detail="Realtime inference engine not available")
    
    try:
        stats = realtime_engine.get_performance_stats()
        return {
            "status": "success",
            "timestamp": time.time(),
            "stats": stats
        }
    except Exception as e:
        logger.error(f"Error getting performance stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get performance stats: {str(e)}")


@app.post("/performance/warmup")
async def warmup_inference_engine(num_requests: int = 10):
    """Warm up the inference engine with dummy requests."""
    if not realtime_engine:
        raise HTTPException(status_code=503, detail="Realtime inference engine not available")
    
    try:
        warmup_time = realtime_engine.warmup(num_warmup_requests=num_requests)
        return {
            "status": "success",
            "warmup_time_seconds": warmup_time,
            "num_requests": num_requests,
            "message": f"Inference engine warmed up with {num_requests} requests in {warmup_time:.2f}s"
        }
    except Exception as e:
        logger.error(f"Error warming up inference engine: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to warm up inference engine: {str(e)}")


@app.get("/performance/cache/stats")
async def get_cache_stats():
    """Get prediction cache statistics."""
    if not realtime_engine:
        raise HTTPException(status_code=503, detail="Realtime inference engine not available")
    
    try:
        cache_stats = realtime_engine.prediction_cache.get_stats()
        return {
            "status": "success",
            "timestamp": time.time(),
            "cache_stats": cache_stats
        }
    except Exception as e:
        logger.error(f"Error getting cache stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get cache stats: {str(e)}")


@app.post("/predict", response_model=InteractionResponse)
async def predict_interaction(request: InteractionRequest):
    """Predict protein-drug interaction using trained model."""
    try:
        # Get protein sequence
        if request.protein.uniprot_id:
            protein_info = protein_data.fetch_protein_by_id(request.protein.uniprot_id)
            protein_sequence = protein_info.get("sequence", "")
            if not protein_sequence:
                raise HTTPException(status_code=404, detail="Protein not found")
        else:
            protein_sequence = request.protein.sequence

        # Get drug SMILES
        if request.drug.chembl_id:
            drug_info = drug_data.fetch_drug_by_chembl_id(request.drug.chembl_id)
            smiles_string = drug_info.get("smiles", "")
            if not smiles_string:
                raise HTTPException(status_code=404, detail="Drug not found")
        else:
            smiles_string = request.drug.smiles

        # Validate inputs
        if not protein_data.validate_protein_sequence(protein_sequence):
            raise HTTPException(status_code=400, detail="Invalid protein sequence")
        if not drug_processor.validate_smiles(smiles_string):
            raise HTTPException(status_code=400, detail="Invalid SMILES string")

        # Use realtime inference engine for optimized predictions
        try:
            if realtime_engine:
                # Use optimized realtime inference
                import time
                start_time = time.time()
                result = await realtime_engine.predict_single(
                    protein_sequence, 
                    smiles_string,
                    f"api_{int(time.time() * 1000000)}"
                )
                processing_time = (time.time() - start_time) * 1000
                
                # Determine binding strength based on affinity
                if result.binding_affinity > 7:
                    binding_strength = "Strong"
                elif result.binding_affinity > 5:
                    binding_strength = "Moderate"
                elif result.binding_affinity > 3:
                    binding_strength = "Weak"
                else:
                    binding_strength = "Very Weak"
                
                return InteractionResponse(
                    interaction_probability=float(result.binding_affinity / 10.0),  # Normalize to 0-1
                    binding_strength=binding_strength,
                    confidence=float(result.confidence),
                    protein_properties={
                        "length": len(protein_sequence),
                        "cached": result.cached,
                        "model_version": result.model_version
                    },
                    drug_properties={"smiles": smiles_string},
                    processing_time=float(processing_time)
                )
            else:
                # Fallback to basic prediction service
                affinity = prediction_service.predict_interaction(protein_sequence, smiles_string)
                return InteractionResponse(
                    interaction_probability=float(affinity / 10.0),  # Normalize to 0-1
                    binding_strength="N/A",
                    confidence=0.9,
                    protein_properties={"length": len(protein_sequence)},
                    drug_properties={"smiles": smiles_string},
                    processing_time=0.0
                )
        except Exception as e:
            logger.error(f"Model prediction failed: {e}")
            raise HTTPException(status_code=500, detail=f"Model prediction failed: {str(e)}")
    except Exception as e:
        logger.error(f"Error in prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/analyze/protein", response_model=ProteinAnalysisResponse)
async def analyze_protein(protein: ProteinInput):
    """Analyze protein sequence."""
    try:
        # Get protein sequence
        if protein.uniprot_id:
            protein_info = protein_data.fetch_protein_by_id(protein.uniprot_id)
            sequence = protein_info.get("sequence", "")
            if not sequence:
                raise HTTPException(status_code=404, detail="Protein not found")
        else:
            sequence = protein.sequence
        
        # Analyze protein
        stats = protein_data.get_sequence_stats(sequence)
        
        if not stats["valid"]:
            raise HTTPException(status_code=400, detail="Invalid protein sequence")
        
        # Get ESM encoding
        encoding = esm_model.encode_protein(sequence)
        
        return ProteinAnalysisResponse(
            sequence_length=stats["length"],
            molecular_weight=stats["molecular_weight"],
            amino_acid_composition=stats["amino_acid_counts"],
            hydrophobic_ratio=stats["hydrophobic_ratio"],
            charged_ratio=stats["charged_ratio"],
            embedding_dimension=list(encoding["pooled_embeddings"].shape),
            valid=True
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error analyzing protein: {e}")
        raise HTTPException(status_code=500, detail=f"Protein analysis failed: {str(e)}")


@app.post("/analyze/drug", response_model=DrugAnalysisResponse)
async def analyze_drug(drug: DrugInput):
    """Analyze drug molecule."""
    try:
        # Get SMILES string
        if drug.chembl_id:
            drug_info = drug_data.fetch_drug_by_chembl_id(drug.chembl_id)
            smiles = drug_info.get("smiles", "")
            if not smiles:
                raise HTTPException(status_code=404, detail="Drug not found")
        else:
            smiles = drug.smiles
        
        # Analyze drug
        analysis = drug_processor.process_smiles(smiles)
        
        if not analysis["valid"]:
            raise HTTPException(status_code=400, detail=f"Invalid SMILES: {analysis.get('error', 'Unknown error')}")
        
        descriptors = analysis["descriptors"]
        drug_likeness = analysis["drug_likeness"]
        
        return DrugAnalysisResponse(
            molecular_weight=descriptors.get("molecular_weight", 0),
            logp=descriptors.get("logp", 0),
            hbd=descriptors.get("hbd", 0),
            hba=descriptors.get("hba", 0),
            tpsa=descriptors.get("tpsa", 0),
            rotatable_bonds=descriptors.get("rotatable_bonds", 0),
            lipinski_compliant=drug_likeness.get("lipinski_compliant", False),
            drug_like_score=drug_likeness.get("drug_like_score", 0),
            valid=True
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error analyzing drug: {e}")
        raise HTTPException(status_code=500, detail=f"Drug analysis failed: {str(e)}")


@app.get("/search/proteins", response_model=SearchResponse)
async def search_proteins(keyword: str, limit: int = 10):
    """Search proteins by keyword."""
    try:
        results = protein_data.search_proteins_by_keyword(keyword, limit)
        
        return SearchResponse(
            results=results,
            total_count=len(results),
            query=keyword
        )
        
    except Exception as e:
        logger.error(f"Error searching proteins: {e}")
        raise HTTPException(status_code=500, detail=f"Protein search failed: {str(e)}")


@app.get("/search/drugs", response_model=SearchResponse)
async def search_drugs(target_id: str, limit: int = 10):
    """Search drugs by target."""
    try:
        results = drug_data.search_drugs_by_target(target_id, limit)
        
        return SearchResponse(
            results=results,
            total_count=len(results),
            query=target_id
        )
        
    except Exception as e:
        logger.error(f"Error searching drugs: {e}")
        raise HTTPException(status_code=500, detail=f"Drug search failed: {str(e)}")


@app.get("/drugs/approved", response_model=SearchResponse)
async def get_approved_drugs(limit: int = 50):
    """Get approved drugs."""
    try:
        results = drug_data.get_approved_drugs(limit)
        
        return SearchResponse(
            results=results,
            total_count=len(results),
            query="approved_drugs"
        )
        
    except Exception as e:
        logger.error(f"Error fetching approved drugs: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch approved drugs: {str(e)}")


# Batch Processing Endpoints
@app.post("/batch/submit", response_model=BatchJobResponse)
async def submit_batch_job(
    request: BatchJobRequest, 
    background_tasks: BackgroundTasks,
    current_user = Depends(get_current_active_user)
):
    """Submit a batch job for protein-drug interaction predictions."""
    try:
        # Validate inputs
        if not request.protein_sequences:
            raise HTTPException(status_code=400, detail="No protein sequences provided")
        if not request.drug_smiles:
            raise HTTPException(status_code=400, detail="No drug SMILES provided")
        
        # Validate sequences and SMILES
        for i, seq in enumerate(request.protein_sequences):
            if not protein_data.validate_protein_sequence(seq):
                raise HTTPException(status_code=400, detail=f"Invalid protein sequence at index {i}")
        
        for i, smiles in enumerate(request.drug_smiles):
            if not drug_processor.validate_smiles(smiles):
                raise HTTPException(status_code=400, detail=f"Invalid SMILES string at index {i}")
        
        # Create batch job
        job_id = batch_service.create_job(request.protein_sequences, request.drug_smiles)
        
        # Process job in background using realtime inference engine if available
        predictor = realtime_engine if realtime_engine else prediction_service
        background_tasks.add_task(batch_service.process_job, job_id, predictor)
        
        return BatchJobResponse(
            job_id=job_id,
            status="submitted",
            message=f"Batch job submitted with {len(request.protein_sequences)} proteins and {len(request.drug_smiles)} drugs"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error submitting batch job: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to submit batch job: {str(e)}")


@app.get("/batch/status/{job_id}", response_model=BatchJobStatus)
async def get_batch_job_status(job_id: str):
    """Get the status of a batch job."""
    try:
        status = batch_service.get_job_status(job_id)
        
        if not status:
            raise HTTPException(status_code=404, detail="Job not found")
        
        return BatchJobStatus(**status)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting job status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get job status: {str(e)}")


@app.get("/batch/jobs")
async def list_batch_jobs():
    """List all batch jobs."""
    try:
        jobs = batch_service.list_jobs()
        return {"jobs": jobs}
        
    except Exception as e:
        logger.error(f"Error listing jobs: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list jobs: {str(e)}")


@app.delete("/batch/jobs/{job_id}")
async def delete_batch_job(job_id: str):
    """Delete a batch job."""
    try:
        success = batch_service.delete_job(job_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="Job not found")
        
        return {"message": f"Job {job_id} deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting job: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete job: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)