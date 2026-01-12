# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Context

This is a personal blogging twin project that fine-tunes llama3.2:1b on the author's blog posts to replicate their writing style. The system has a modular CLI architecture where components can be run individually or as a pipeline, plus a React web interface for interaction.

## Model Configuration

**Critical**: All model interactions use this specific configuration:

```python
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

model = OpenAIModel(
    model_name="llama3.2:1b",  # or "blogging-twin:latest" for fine-tuned
    provider=OpenAIProvider(base_url="http://localhost:11434/v1")
)
```

The base model is `llama3.2:1b` hosted locally via Ollama. After fine-tuning, the model is deployed as `blogging-twin:latest` (or versioned like `blogging-twin:v1`).

## Dependency Management

This project uses **uv** for modern Python dependency management:
- **Source of truth**: `pyproject.toml` (NOT requirements.txt)
- **Lock file**: `uv.lock` ensures reproducible builds (commit to git)
- **Virtual environment**: `.venv/` (auto-created by `uv sync`)

To add dependencies:
```bash
uv add package-name              # Add to dependencies
uv add --dev package-name        # Add to dev dependencies
uv sync                          # Install/update all dependencies
```

## Common Commands

### Setup
```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install Python dependencies (creates .venv and uv.lock)
uv sync

# Install Ollama and pull base model
curl https://ollama.ai/install.sh | sh
ollama pull llama3.2:1b

# Setup frontend
cd frontend
npm install
```

### Running the System
```bash
# Verify Ollama is running
ollama list
OR visit http://localhost:11434 - you should see "Ollama is running"

# Start backend API server (uses uv to run with virtual environment)
uv run python cli.py serve --port 8000
And test in this http://localhost:8000/ - you will see a similar version - {"message":"LLM Blogging Twin API","version":"1.0.0","docs":"/docs","health":"/health"}

# Start frontend (separate terminal)
cd frontend
npm run dev
And view in this http://localhost:5173/

```

### CLI Pipeline Commands
```bash
# Individual steps
uv run python cli.py ingest --input-dir ./data/raw --output ./data/processed/raw_blogs.jsonl
uv run python cli.py prepare-dataset --input ./data/processed/raw_blogs.jsonl --output ./data/processed/training_data.jsonl
uv run python cli.py finetune --data ./data/processed/training_data.jsonl --output ./models/blogging_twin
uv run python cli.py deploy --model-path ./models/blogging_twin --model-name blogging-twin:v1
uv run python cli.py evaluate --model blogging-twin:v1

# Full pipeline (runs all steps)
uv run python cli.py pipeline --blog-dir ./data/raw --model-name blogging-twin:latest
```

### Steps to Fine-tune the Model
- There are two approaches: individual steps (recommended for learning/debugging) or full pipeline (automated).

####  Option 1: Individual Steps (Recommended First Time)
- Step 1: Prepare Your Blog Data
  - Place your blog files (.md, .txt, or .docx) in the data directory:
  - mkdir -p data/raw
  - Copy your blog files to data/raw/

- Step 2: Ingest the Blogs
  - Convert blog files to JSONL format:
  - uv run python cli.py ingest --input-dir ./data/raw --output ./data/processed/raw_blogs.jsonl

- Step 3: Prepare Training Dataset
  - Convert to instruction-response format for fine-tuning:
  - uv run python cli.py prepare-dataset --input ./data/processed/raw_blogs.jsonl --output ./data/processed/training_data.jsonl

- Step 4: Fine-tune the Model
  - Train the model with LoRA adapters:
  - uv run python cli.py finetune --data ./data/processed/training_data.jsonl --output ./models/blogging_twin

- Step 5: Deploy to Ollama
  - Convert and import the fine-tuned model:
  - uv run python cli.py deploy --model-path ./models/blogging_twin --model-name blogging-twin:v1

- Step 6: Evaluate (Optional)
  - Test the fine-tuned model:
  - uv run python cli.py evaluate --model blogging-twin:v1

####  Option 2: Full Pipeline (Automated)
- Run everything in one command:
- uv run python cli.py pipeline --blog-dir ./data/raw --model-name blogging-twin:latest
- This runs all steps automatically: ingest → prepare → finetune → deploy → evaluate.



## Architecture Overview

### Three-Tier System
1. **CLI Layer** (`cli.py`): Command orchestration and pipeline execution
2. **Processing Layer** (`src/`): Data ingestion, fine-tuning, deployment, evaluation
3. **Interface Layer**: FastAPI backend (`api/`) + React frontend (`frontend/`)

### Data Flow Pattern
All data transformations follow this pattern:
- Raw blog files (.md/.txt/.docx) → JSONL format → Training format → Model weights → Ollama deployment

The training data format is instruction-response pairs:
```json
{
  "instruction": "Write a blog post about [topic]",
  "input": "",
  "output": "[actual blog content]"
}
```

### Model Lifecycle
1. **Base model**: `llama3.2:1b` pulled from Ollama registry
2. **Fine-tuning**: Uses unsloth.ai with LoRA adapters
3. **Deployment**: Converted to Ollama Modelfile format and imported
4. **Serving**: Both base and fine-tuned models available via OpenAI-compatible API

## Key Technical Decisions

### Why Pydantic AI with OpenAI Provider
We use pydantic-ai's OpenAI provider pointed at Ollama's local endpoint (`http://localhost:11434/v1`) rather than Ollama's native Python library. This provides:
- Standardized OpenAI API interface
- Easy model switching (base vs. fine-tuned)
- Consistent with pydantic-ai patterns

### Why Unsloth for Fine-tuning
Unsloth.ai is chosen for:
- Efficient LoRA implementation
- Memory optimization for local training
- Easy export to Ollama-compatible formats

### Modular CLI Design
Each component (`ingest`, `prepare-dataset`, `finetune`, `deploy`, `evaluate`) is:
- Independently executable for development/debugging
- Composable in the pipeline command
- Configured via `pipeline_config.yaml` for consistency

## Frontend-Backend Communication

The React frontend communicates with FastAPI backend via three main endpoints:

**POST `/api/generate`**: Generate text from selected model
```javascript
{
  "prompt": "Write about AI safety",
  "model": "base" | "finetuned",  // which model to use
  "max_tokens": 500,
  "temperature": 0.7
}
```

**GET `/api/models`**: Returns available models and their status
```javascript
{
  "base": { "name": "llama3.2:1b", "available": true },
  "finetuned": { "name": "blogging-twin:latest", "available": true }
}
```

**POST `/api/evaluate`**: Side-by-side comparison (generates from both models)

The frontend displays which model is currently selected in the UI (as shown in the design mockup with model dropdown).

## File Organization Logic

### Data Directory Structure
```
data/
├── raw/                  # Original blog files go here
├── processed/            # Output from ingestion (JSONL)
└── validation/           # Hold-out test set for evaluation
```

### Models Directory
```
models/
└── checkpoints/          # Saved model weights and adapters
```

Each training run should create a versioned subdirectory (e.g., `models/blogging_twin_v1/`) containing:
- LoRA adapter weights
- Training logs
- Model configuration

### Results Directory
```
results/
└── evaluations/          # Evaluation reports (JSON)
```

## Configuration Management

All pipeline parameters are centralized in `pipeline_config.yaml`:
- Data paths
- Training hyperparameters (epochs, learning rate, LoRA config)
- Deployment settings (model name, Ollama URL)
- Evaluation options

When modifying training behavior, update the config file rather than hardcoding values.

## Important Patterns

### CLI Module Pattern
Each CLI command maps to a function in `src/`:
- `cli.py ingest` → `src/data_ingestion.py:run_ingestion()`
- `cli.py finetune` → `src/finetune.py:run_finetuning()`

This separation keeps CLI argument parsing separate from business logic.

### Model Selection in Backend
The backend maintains two initialized model instances:
```python
base_model = OpenAIModel(model_name="llama3.2:1b", ...)
finetuned_model = OpenAIModel(model_name="blogging-twin:latest", ...)
```

Requests specify which model to use via the `model` parameter.

### Error Handling in Pipeline
The pipeline should:
- Validate inputs before processing (file existence, format)
- Check Ollama availability before deployment
- Gracefully handle missing models (fall back to base model)
- Log all errors to `pipeline.log`

## Development Workflow

### Adding New Blog Files
1. Place files in `data/raw/`
2. Run `uv run python cli.py ingest` to process them
3. Verify output in `data/processed/raw_blogs.jsonl`

### Retraining the Model
1. Update blogs in `data/raw/`
2. Run full pipeline: `uv run python cli.py pipeline --blog-dir ./data/raw --model-name blogging-twin:v2`
3. Compare old vs. new model using evaluation

### Testing Model Changes
1. Make changes to fine-tuning parameters in `pipeline_config.yaml`
2. Run `uv run python cli.py finetune` (uses existing prepared data)
3. Deploy with a new name: `uv run python cli.py deploy --model-name blogging-twin:test`
4. Compare in UI by adding to model list

## Evaluation Philosophy

The evaluation module should answer:
1. **Does it sound like the author?** (style similarity metrics)
2. **Is it coherent?** (perplexity, readability)
3. **Is it better than base?** (side-by-side comparison)

Focus evaluation on qualitative assessment (human review of sample outputs) over purely quantitative metrics. The test prompts should cover diverse topics the author typically writes about.

## Frontend Design Guidelines

The UI is inspired by claude.ai (see design mockup):
- Clean, minimalist greeting interface
- Prominent model selector showing current model
- Suggested prompt categories as quick-start options
- Clear visual distinction between base and fine-tuned model outputs in comparison mode

**Responsive Design**:
- **Flexbox layout** (`flex flex-col min-h-screen`) ensures footer stays at bottom
- **Dynamic viewport calculations**: ChatInterface uses `calc(90vh - 280px)` for responsive height
- **Fixed footer**: Positioned with `mt-auto` in flexbox container for consistent bottom placement
- **Two-mode interface**:
  - Greeting: Centered layout with input and suggested prompts
  - Conversation: Scrollable messages with fixed input at bottom
- **Mobile-friendly**: Layout adapts to different screen sizes using Tailwind responsive utilities

When implementing new UI features, maintain this simplicity and avoid adding unnecessary complexity. Always test responsive behavior across viewport sizes.
