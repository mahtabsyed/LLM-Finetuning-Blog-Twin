# Requirements and Design Document
## LLM Blogging Twin - Fine-tuning Project

### Project Overview
Create a personal blogging twin by fine-tuning the llama3.2:1b model on previously written blog posts. The system will learn the author's writing style and voice to generate blog content that matches their unique perspective.

### Core Objectives
1. Fine-tune llama3.2:1b to replicate personal blogging style
2. Create a modular, CLI-driven architecture with pipeline capability
3. Build an interactive web interface for model interaction
4. Enable comparison between base and fine-tuned models
5. Provide evaluation metrics to assess fine-tuning quality

---

## System Architecture

### Technology Stack

#### Backend
- **Python**: Core language for ML pipeline
- **Ollama**: Local LLM hosting (llama3.2:1b)
- **pydantic-ai**: Model interaction via OpenAI-compatible interface
- **unsloth.ai / unsloth-mlx**: Fine-tuning framework (platform-specific)
  - **unsloth**: For NVIDIA/AMD/Intel GPUs (CUDA/ROCm)
  - **unsloth-mlx**: For Apple Silicon Macs (MLX framework)
- **FastAPI**: Backend API server (for frontend communication)

#### Frontend
- **React.js**: Web interface
- **Axios/Fetch**: API communication
- **TailwindCSS**: Styling (based on claude.ai-like design)

#### Data Processing
- **python-docx**: DOCX file parsing
- **markdown**: Markdown file parsing
- **pandas**: Data manipulation and dataset creation

---

## Component Architecture

### 1. Data Ingestion Module (`data_ingestion.py`)

**Purpose**: Read and parse blog files from local storage

**Inputs**:
- Directory path containing blog files
- Supported formats: `.md`, `.txt`, `.docx`

**Outputs**:
- Structured dataset (JSON/JSONL format)
- Metadata: filename, word count, creation date

**Key Functions**:
```python
def read_blogs(directory_path: str) -> list[dict]
def parse_markdown(file_path: str) -> str
def parse_docx(file_path: str) -> str
def parse_txt(file_path: str) -> str
```

**CLI Command**:
```bash
python cli.py ingest --input-dir ./blogs --output ./data/raw_blogs.jsonl
```

---

### 2. Dataset Preparation Module (`dataset_prep.py`)

**Purpose**: Transform raw blog data into fine-tuning format

**Processing Steps**:
1. Clean and normalize text
2. Create instruction-response pairs
3. Format data for unsloth.ai fine-tuning
4. Split into train/validation sets

**Output Format**:
```json
{
  "instruction": "Write a blog post about [topic]",
  "input": "",
  "output": "[actual blog content]"
}
```

**CLI Command**:
```bash
python cli.py prepare-dataset --input ./data/raw_blogs.jsonl --output ./data/training_data.jsonl --split 0.8
```

---

### 3. Fine-tuning Module (`finetune.py`)

**Purpose**: Fine-tune llama3.2:1b using platform-appropriate framework

**Platform-Aware Architecture**:
The fine-tuning module automatically detects hardware and selects the appropriate library:

| Platform | Library | Framework | Model Repository |
|----------|---------|-----------|------------------|
| Apple Silicon (M1/M2/M3/M4/M5) | unsloth-mlx | MLX | mlx-community/Llama-3.2-1B-Instruct-4bit |
| NVIDIA GPU | unsloth | CUDA | unsloth/llama-3.2-1b |
| AMD GPU | unsloth | ROCm | unsloth/llama-3.2-1b |
| Intel GPU | unsloth | XPU | unsloth/llama-3.2-1b |

**Configuration**:
- Base model: `llama3.2:1b` (auto-mapped to platform-specific variant)
- Fine-tuning method: LoRA (Low-Rank Adaptation)
- Training parameters: Learning rate, epochs, batch size
- Hardware detection: Automatic via `detect_hardware_platform()`

**Process**:
1. Detect hardware platform (Apple Silicon vs CUDA vs CPU)
2. Import appropriate library (`unsloth_mlx` or `unsloth`)
3. Load platform-specific base model
4. Apply LoRA adapters
5. Configure platform-aware training arguments
6. Train on prepared dataset
7. Save fine-tuned weights
8. Export to Ollama-compatible format

**Key Functions**:
```python
def detect_hardware_platform() -> str:
    """Returns 'apple_silicon', 'cuda', or 'cpu'"""

def get_platform_model_name(platform: str, base_model: str) -> str:
    """Maps generic model name to platform-specific repository"""

def load_base_model(...):
    """Dynamically imports and loads appropriate FastLanguageModel"""

def train_model(...):
    """Platform-aware training configuration (MLX skips precision flags)"""
```

**CLI Command**:
```bash
uv run python cli.py finetune --data ./data/training_data.jsonl --output ./models/blogging_twin --epochs 3 --lr 2e-4
```

---

### 4. Model Deployment Module (`deploy.py`)

**Purpose**: Deploy fine-tuned model to local Ollama instance

**Steps**:
1. Convert fine-tuned weights to GGUF format (if needed)
2. Create Ollama Modelfile
3. Import model to Ollama
4. Verify model availability

**CLI Command**:
```bash
python cli.py deploy --model-path ./models/blogging_twin --model-name blogging-twin:latest
```

**Ollama Modelfile Template**:
```
FROM llama3.2:1b
ADAPTER ./blogging_twin_adapter
PARAMETER temperature 0.7
PARAMETER top_p 0.9
```

---

### 5. Backend API (`api/server.py`)

**Purpose**: FastAPI server to connect frontend with Ollama models

**Endpoints**:

```python
POST /api/generate
{
  "prompt": str,
  "model": "base" | "finetuned",
  "max_tokens": int,
  "temperature": float
}

GET /api/models
# Returns available models and their status

POST /api/evaluate
{
  "prompt": str
}
# Generates responses from both models for comparison
```

**Implementation**:
```python
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

base_model = OpenAIModel(
    model_name="llama3.2:1b",
    provider=OpenAIProvider(base_url="http://localhost:11434/v1")
)

finetuned_model = OpenAIModel(
    model_name="blogging-twin:latest",
    provider=OpenAIProvider(base_url="http://localhost:11434/v1")
)
```

**CLI Command**:
```bash
python cli.py serve --port 8000
```

---

### 6. Frontend Web Interface (`frontend/`)

**Purpose**: Interactive chat interface for model interaction

**Design Reference**: Claude.ai-style interface (from screenshot)

**Features**:
- Clean, minimalist design with greeting message
- Input prompt with attachment button
- Model selector dropdown (showing current model)
- Suggested prompts/categories (Write, Learn, Code, Life stuff, Claude's choice)
- Response display area
- Model comparison mode (side-by-side output)
- Responsive layout with flexbox design
- Persistent footer that stays visible at the bottom
- Conversation-based UI (centered greeting → scrollable chat with fixed input)

**Key Components**:
```
frontend/
├── src/
│   ├── components/
│   │   ├── ChatInterface.jsx
│   │   ├── ModelSelector.jsx
│   │   ├── PromptInput.jsx
│   │   └── ComparisonView.jsx
│   ├── services/
│   │   └── api.js
│   └── App.jsx
├── public/
└── package.json
```

**API Integration**:
```javascript
const generateResponse = async (prompt, model) => {
  const response = await fetch('http://localhost:8000/api/generate', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ prompt, model })
  });
  return await response.json();
};
```

**Run Commands**:
```bash
cd frontend
npm install
npm run dev
```

**Responsive Design Implementation**:
- **Flexbox layout**: Uses `flex flex-col` for vertical stacking with responsive footer
- **Dynamic height calculation**: ChatInterface height adjusts based on viewport (`calc(90vh - 280px)`)
- **Fixed footer**: Always visible at bottom using `mt-auto` in flexbox container
- **Viewport-responsive**: Works across different screen sizes (mobile, tablet, desktop)
- **Two-mode layout**:
  - Greeting mode: Centered content with input box and suggested prompts
  - Conversation mode: Scrollable message area with fixed input at bottom

---

### 7. Evaluation Module (`evaluate.py`)

**Purpose**: Assess fine-tuning quality and model performance

**Metrics**:
1. **Perplexity**: Measure model confidence
2. **Style Similarity**: Compare writing patterns
3. **Human Evaluation**: Side-by-side comparison prompts
4. **BLEU/ROUGE Scores**: Content overlap with original blogs
5. **Response Quality**: Coherence, relevance, creativity

**Evaluation Types**:
- **Automated**: Run on validation set
- **Interactive**: Generate responses to test prompts
- **Comparative**: Base vs. fine-tuned model outputs

**CLI Command**:
```bash
python cli.py evaluate --model blogging-twin:latest --test-set ./data/validation.jsonl --output ./results/eval_report.json
```

**Output**:
```json
{
  "perplexity": 12.3,
  "style_similarity_score": 0.85,
  "average_bleu": 0.42,
  "sample_comparisons": [...]
}
```

---

### 8. Pipeline Orchestration (`pipeline.py`)

**Purpose**: Run complete workflow from ingestion to deployment

**Pipeline Stages**:
1. Data ingestion
2. Dataset preparation
3. Fine-tuning
4. Deployment
5. Evaluation

**CLI Command**:
```bash
python cli.py pipeline --blog-dir ./blogs --model-name blogging-twin:latest
```

**Configuration File** (`pipeline_config.yaml`):
```yaml
data:
  input_dir: ./blogs
  output_dir: ./data
  formats: [md, txt, docx]

training:
  epochs: 3
  learning_rate: 2e-4
  batch_size: 4
  lora_r: 16
  lora_alpha: 32

deployment:
  model_name: blogging-twin:latest
  ollama_base_url: http://localhost:11434

evaluation:
  run_after_training: true
  test_prompts_file: ./prompts/test_prompts.txt
```

---

## CLI Structure

### Main CLI (`cli.py`)

```
Commands:
  ingest            Read and parse blog files
  prepare-dataset   Create training data format
  finetune          Fine-tune llama3.2:1b model
  deploy            Deploy model to Ollama
  serve             Start FastAPI backend server
  evaluate          Run model evaluation
  pipeline          Execute full training pipeline
```

**Example Usage**:
```bash
# Individual steps
python cli.py ingest --input-dir ./blogs --output ./data/raw.jsonl
python cli.py prepare-dataset --input ./data/raw.jsonl --output ./data/train.jsonl
python cli.py finetune --data ./data/train.jsonl --output ./models/v1
python cli.py deploy --model-path ./models/v1 --model-name blogging-twin:v1
python cli.py evaluate --model blogging-twin:v1

# Full pipeline
python cli.py pipeline --blog-dir ./blogs --model-name blogging-twin:v1

# Start web interface
python cli.py serve --port 8000
# In another terminal:
cd frontend && npm run dev
```

---

## Data Flow

```
[Blog Files (.md/.txt/.docx)]
          ↓
    [Data Ingestion]
          ↓
   [Raw JSONL Dataset]
          ↓
  [Dataset Preparation]
          ↓
  [Training JSONL Format]
          ↓
   [Fine-tuning (unsloth)]
          ↓
  [Fine-tuned Model Weights]
          ↓
    [Ollama Deployment]
          ↓
  [Local Model Serving]
          ↓
    [FastAPI Backend] ←→ [React Frontend]
          ↓
   [User Interaction]
```

---

## Project Structure

```
llm-blogging-twin/
├── README.md
├── REQUIREMENTS_AND_DESIGN.md
├── CLAUDE.md
├── cli.py                      # Main CLI entry point
├── requirements.txt
├── pipeline_config.yaml
│
├── src/
│   ├── __init__.py
│   ├── data_ingestion.py       # Blog file reading
│   ├── dataset_prep.py         # Dataset formatting
│   ├── finetune.py             # Unsloth fine-tuning
│   ├── deploy.py               # Ollama deployment
│   ├── evaluate.py             # Model evaluation
│   └── pipeline.py             # Orchestration logic
│
├── api/
│   ├── __init__.py
│   ├── server.py               # FastAPI backend
│   └── models.py               # Pydantic models
│
├── frontend/
│   ├── package.json
│   ├── src/
│   │   ├── App.jsx
│   │   ├── components/
│   │   └── services/
│   └── public/
│
├── data/
│   ├── raw/                    # Raw blog files
│   ├── processed/              # Processed datasets
│   └── validation/             # Test sets
│
├── models/
│   └── checkpoints/            # Saved model weights
│
├── prompts/
│   └── test_prompts.txt        # Evaluation prompts
│
└── results/
    └── evaluations/            # Evaluation reports
```

---

## Development Workflow

### Phase 1: Data Preparation
1. Collect blog files in `data/raw/`
2. Run ingestion: `python cli.py ingest`
3. Prepare dataset: `python cli.py prepare-dataset`
4. Review and validate training data

### Phase 2: Model Fine-tuning
1. Configure `pipeline_config.yaml`
2. Run fine-tuning: `python cli.py finetune`
3. Monitor training metrics
4. Save checkpoints

### Phase 3: Deployment
1. Deploy to Ollama: `python cli.py deploy`
2. Verify model: `ollama list`
3. Test basic inference: `ollama run blogging-twin:latest "Write about AI"`

### Phase 4: Web Interface
1. Start backend: `python cli.py serve`
2. Start frontend: `cd frontend && npm run dev`
3. Test interaction at `http://localhost:5173`

### Phase 5: Evaluation
1. Run automated evaluation: `python cli.py evaluate`
2. Compare base vs. fine-tuned outputs
3. Iterate on training if needed

---

## Key Dependencies

### Python Package Management

This project uses **uv** for modern Python dependency management:
- **Source of truth**: `pyproject.toml` (NOT requirements.txt)
- **Lock file**: `uv.lock` ensures reproducible builds (committed to git)
- **Virtual environment**: `.venv/` (auto-created by `uv sync`)

### Python (`pyproject.toml`)
```toml
[project]
name = "llm-blogging-twin"
version = "1.0.0"
requires-python = ">=3.10"

dependencies = [
    "fastapi>=0.104.0",
    "uvicorn[standard]>=0.24.0",
    "pydantic-ai>=0.0.13",
    "unsloth[colab-new]>=2024.1",
    "torch>=2.1.0",
    "transformers>=4.35.0",
    "python-docx>=1.1.0",
    "markdown>=3.5.0",
    "pandas>=2.1.0",
    "pyyaml>=6.0",
    "click>=8.1.7",
    "requests>=2.31.0",
]

[dependency-groups]
dev = [
    "pytest>=7.4.0",
    "black>=23.0.0",
    "flake8>=6.0.0",
]
```

**Installation**:
```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies (creates .venv and uv.lock)
uv sync

# Activate virtual environment
source .venv/bin/activate  # On macOS/Linux
```

### Frontend (`frontend/package.json`)
```json
{
  "dependencies": {
    "react": "^18.2.0",
    "react-dom": "^18.2.0",
    "axios": "^1.6.0",
    "tailwindcss": "^3.3.0"
  }
}
```

---

## Model Hosting Requirements

### Ollama Setup
```bash
# Install Ollama
curl https://ollama.ai/install.sh | sh

# Pull base model
ollama pull llama3.2:1b

# Verify installation
ollama list
```

### System Requirements
- **RAM**: Minimum 8GB (16GB recommended)
- **Storage**: 10GB for models and data
- **GPU**: Optional but recommended (CUDA-compatible)

---

## Evaluation Strategy

### Test Prompts
Create diverse prompts covering different blog topics:
- Technical tutorials
- Personal reflections
- Opinion pieces
- How-to guides
- Commentary on current events

### Comparison Framework
For each prompt, generate:
1. Base model response
2. Fine-tuned model response
3. Original blog excerpt (if applicable)

### Quality Metrics
- **Voice consistency**: Does it sound like the author?
- **Topic relevance**: Stays on subject
- **Creativity**: Generates unique perspectives
- **Coherence**: Logical flow and structure
- **Length appropriateness**: Matches blog post length patterns

---

## Future Enhancements

1. **Multi-user support**: Fine-tune for multiple authors
2. **Continuous learning**: Periodically retrain on new blogs
3. **Style transfer controls**: Adjust formality, tone, etc.
4. **Blog drafting assistant**: Collaborative writing interface
5. **Analytics dashboard**: Track model performance over time
6. **Export functionality**: Save generated content as markdown/HTML
