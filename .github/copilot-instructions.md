# Copilot Instructions for Protein-Drug Discovery Codebase

## Overview
This repository implements a modular protein-drug discovery system. Major components are organized under `protein_drug_discovery/`:
- **core/**: Model management, training workflows, prediction engines
- **data/**: Data loaders, preprocessing, and dataset utilities
- **models/**: Encoders, predictors, and training scripts
- **ui/**: Streamlit-based user interface
- **visualization/**: Tools for binding and structural analysis
- **api/**: REST API entry point

## Key Workflows
- **Training**: Use scripts in `scripts/` (e.g., `train_standard_model.py`, `train_unsloth_model.py`). Models and logs are saved in `models/` and `logs/`.
- **Testing**: Run system and unit tests with:
  - `python -m pytest tests/` (all tests)
  - `python tests/test_complete_system.py` (end-to-end)
  - `python test_clean_system.py` (clean model)
- **API**: Start with `python scripts/run_api.py`.
- **UI**: Launch Streamlit app via `python scripts/run_streamlit.py`.

## Patterns & Conventions
- **Data Flow**: Data is loaded from `datasets/` via loaders in `protein_drug_discovery/data/`. Models consume processed data from these loaders.
- **Model Management**: All model logic is centralized in `core/model_manager.py` and related files. Training workflows are in `core/training_workflow.py`.
- **Testing**: Tests are organized by type in `tests/` and as standalone scripts. Follow existing test file structure for new tests.
- **Configuration**: Adapter and model configs are stored in `test_lora_adapter/`.
- **Logging**: Output and logs are written to `logs/`.

## Integration Points
- **External Dependencies**: All requirements are listed in `requirements.txt`. Install with `pip install -r requirements.txt`.
- **Cross-Component Communication**: Core modules import from `data/` and `models/` using relative imports. UI and API interact with core via exposed interfaces.

## Examples
- To train a model: `python scripts/train_standard_model.py`
- To run all tests: `python -m pytest tests/`
- To launch the UI: `python scripts/run_streamlit.py`

## Tips for AI Agents
- Always check `README.md` and `docs/README.md` for workflow details and examples.
- When adding new features, update or add tests in `tests/` and scripts in `scripts/`.
- Follow the modular structure—keep new logic in the appropriate submodule.
- Use existing data loaders and model managers as templates for new components.

---
For questions or unclear conventions, review the latest `README.md` or ask for clarification.
