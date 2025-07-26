"""Core components for protein-drug discovery."""

from .esm_model import ESMProteinModel
from .drug_processor import DrugProcessor

# Import trainers with error handling
try:
    from .standard_trainer import StandardProteinTrainer
    STANDARD_TRAINER_AVAILABLE = True
except ImportError:
    STANDARD_TRAINER_AVAILABLE = False

try:
    from .unsloth_trainer import UnslothProteinTrainer
    UNSLOTH_TRAINER_AVAILABLE = True
except ImportError:
    UNSLOTH_TRAINER_AVAILABLE = False

__all__ = ["ESMProteinModel", "DrugProcessor"]

if STANDARD_TRAINER_AVAILABLE:
    __all__.append("StandardProteinTrainer")
if UNSLOTH_TRAINER_AVAILABLE:
    __all__.append("UnslothProteinTrainer")