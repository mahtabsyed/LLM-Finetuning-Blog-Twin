# LLM Blogging Twin - Fine-tune Your Personal AI Writing Assistant

> **Learn by doing**: This project teaches you how to fine-tune a local LLM to mimic your writing style. Perfect for understanding LLM fine-tuning, model deployment, and building AI-powered applications.

![Project Type](https://img.shields.io/badge/Type-Educational-blue)
![Python](https://img.shields.io/badge/Python-3.10+-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

## What This Project Does

This project fine-tunes the **llama3.2:1b** model on your personal blog posts to create an AI that writes like you. It includes:

- 📚 **Data ingestion** from markdown, text, and Word documents
- 🔧 **Fine-tuning pipeline** using unsloth.ai for efficient training
- 🚀 **Local deployment** with Ollama (no cloud required!)
- 🌐 **Web interface** to interact with your AI twin
- 📊 **Evaluation tools** to compare base vs. fine-tuned models
- 🔨 **Modular CLI** - run individual steps or complete pipeline

## Why This Project is Great for Learning

### You'll Learn:
1. **LLM Fine-tuning Fundamentals**
   - How to prepare training data for language models
   - LoRA (Low-Rank Adaptation) technique for efficient fine-tuning
   - Using unsloth.ai for optimized training

2. **Local LLM Deployment**
   - Setting up Ollama for local model hosting
   - Working with OpenAI-compatible APIs
   - Model management and version control

3. **Full-Stack AI Application**
   - Building a FastAPI backend for LLM interaction
   - Creating a React frontend for AI chat
   - Integrating frontend and backend with REST APIs

4. **MLOps Practices**
   - Building modular, reusable ML pipelines
   - CLI design for ML workflows
   - Configuration management with YAML

5. **Practical AI Engineering**
   - Data preprocessing for NLP tasks
   - Model evaluation and comparison
   - Production-ready code structure

## Project Architecture

```
┌─────────────────┐
│   Your Blogs    │
│  (.md/.txt/.docx)│
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Data Ingestion  │ ← Parse and clean blog files
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│Dataset Prep     │ ← Format for fine-tuning
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Fine-tuning    │ ← Train with unsloth.ai
│  (unsloth.ai)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│Ollama Deployment│ ← Deploy locally
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────┐
│   FastAPI Backend + React UI    │ ← Interact with your twin
└─────────────────────────────────┘
```

## Quick Start

### Prerequisites

```bash
# System requirements
- Python 3.10 or higher
- Node.js 18+ (for frontend)
- 8GB+ RAM (16GB recommended)
- 10GB free disk space
- uv (Python package manager - will be installed in setup)
```

**Note**: This project uses modern Python packaging with `pyproject.toml` and `uv` for dependency management. The `requirements.txt` file is kept for reference, but `pyproject.toml` is the source of truth.

## Platform Compatibility

This project supports fine-tuning on multiple hardware platforms:

- ✅ **Apple Silicon Macs** (M1/M2/M3/M4/M5) - Uses MLX framework
- ✅ **NVIDIA GPUs** - Uses CUDA
- ✅ **AMD GPUs** - Uses ROCm (via unsloth)
- ✅ **Intel GPUs** - Uses XPU (via unsloth)

### Installation

The correct fine-tuning library is installed automatically based on your hardware:

```bash
# Works on all platforms
uv sync
```

**What gets installed**:
- **Apple Silicon**: `unsloth-mlx` + `mlx`
- **Other platforms**: `unsloth` with CUDA/ROCm support

### Hardware Requirements

**Apple Silicon**:
- macOS 13.0+ (15.0+ recommended for larger models)
- Minimum 16GB unified RAM (32GB+ for models >7B)
- Python 3.10+

**NVIDIA GPU**:
- CUDA 11.8+
- Minimum 8GB VRAM (16GB+ recommended)
- Python 3.10+

### 1. Clone and Setup

```bash
git clone https://github.com/yourusername/llm-blogging-twin.git
cd llm-blogging-twin

# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install all dependencies
# This reads pyproject.toml and creates uv.lock for reproducible builds
uv sync

# Activate the virtual environment
source .venv/bin/activate  # On macOS/Linux
# or: .venv\Scripts\activate  # On Windows

# Install Ollama (macOS/Linux)
curl https://ollama.ai/install.sh | sh

# Pull base model
ollama pull llama3.2:1b

# Verify installation
ollama list
```

### 2. Prepare Your Data

Place your blog files in the `data/raw/` directory:

```bash
mkdir -p data/raw
# Copy your .md, .txt, or .docx blog files to data/raw/
```

### 3. Run the Pipeline

```bash
# Option 1: Run complete pipeline (recommended for first time)
python cli.py pipeline --blog-dir ./data/raw --model-name blogging-twin:v1

# Option 2: Run individual steps (for learning/debugging)
python cli.py ingest --input-dir ./data/raw --output ./data/raw_blogs.jsonl
python cli.py prepare-dataset --input ./data/raw_blogs.jsonl --output ./data/training_data.jsonl
python cli.py finetune --data ./data/training_data.jsonl --output ./models/blogging_twin
python cli.py deploy --model-path ./models/blogging_twin --model-name blogging-twin:v1
python cli.py evaluate --model blogging-twin:v1
```

### 4. Launch the Web Interface

```bash
# Terminal 1: Start backend API
python cli.py serve --port 8000

# Terminal 2: Start frontend
cd frontend
npm install
npm run dev
```

Visit `http://localhost:5173` to interact with your AI blogging twin!

## Project Structure

```
llm-blogging-twin/
├── README.md                    # You are here!
├── REQUIREMENTS_AND_DESIGN.md   # Detailed design docs
├── CLAUDE.md                    # Context for Claude Code
├── pyproject.toml               # Project metadata & dependencies (source of truth)
├── uv.lock                      # Locked dependencies (auto-generated, commit to git)
├── requirements.txt             # Legacy format (for reference)
├── cli.py                       # Main CLI entry point
├── pipeline_config.yaml         # Configuration file
│
├── src/                         # Core modules
│   ├── __init__.py
│   ├── data_ingestion.py       # Read blog files
│   ├── dataset_prep.py         # Format for training
│   ├── finetune.py             # Unsloth fine-tuning
│   ├── deploy.py               # Deploy to Ollama
│   ├── evaluate.py             # Model evaluation
│   └── pipeline.py             # Orchestration
│
├── api/                         # Backend API
│   ├── __init__.py
│   ├── server.py               # FastAPI server
│   └── models.py               # Pydantic schemas
│
├── frontend/                    # React web interface
│   ├── package.json
│   ├── src/
│   │   ├── App.jsx
│   │   ├── components/
│   │   └── services/
│   └── public/
│
├── data/                        # Data directory
│   ├── raw/                    # Your blog files go here
│   ├── processed/              # Processed datasets
│   └── validation/             # Test sets
│
├── models/                      # Model artifacts
│   └── checkpoints/
│
├── prompts/                     # Test prompts
│   └── test_prompts.txt
│
└── results/                     # Evaluation results
    └── evaluations/
```

## Usage Examples

### CLI Commands

```bash
# Ingest blog files
python cli.py ingest --input-dir ./data/raw --output ./data/raw_blogs.jsonl

# Prepare training dataset with 80/20 split
python cli.py prepare-dataset \
  --input ./data/raw_blogs.jsonl \
  --output ./data/training_data.jsonl \
  --split 0.8

# Fine-tune the model
python cli.py finetune \
  --data ./data/training_data.jsonl \
  --output ./models/blogging_twin \
  --epochs 3 \
  --lr 2e-4

# Deploy to Ollama
python cli.py deploy \
  --model-path ./models/blogging_twin \
  --model-name blogging-twin:v1

# Evaluate model performance
python cli.py evaluate \
  --model blogging-twin:v1 \
  --test-set ./data/validation.jsonl \
  --output ./results/eval_report.json

# Start API server
python cli.py serve --port 8000

# Run full pipeline
python cli.py pipeline \
  --blog-dir ./data/raw \
  --model-name blogging-twin:latest
```

### Python API Usage

```python
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

# Connect to your fine-tuned model
model = OpenAIModel(
    model_name="blogging-twin:latest",
    provider=OpenAIProvider(base_url="http://localhost:11434/v1")
)

# Generate blog content
response = model.generate("Write about the future of AI")
print(response)
```

## Configuration

Edit `pipeline_config.yaml` to customize:

```yaml
data:
  input_dir: ./data/raw
  formats: [md, txt, docx]
  train_split: 0.8

training:
  epochs: 3
  learning_rate: 2e-4
  batch_size: 4
  lora_r: 16        # LoRA rank
  lora_alpha: 32    # LoRA alpha

deployment:
  model_name: blogging-twin:latest
  ollama_base_url: http://localhost:11434

evaluation:
  run_after_training: true
  metrics: [perplexity, style_similarity, bleu]
```

## How It Works

### 1. Data Ingestion (`src/data_ingestion.py`)
Reads your blog files and converts them to a structured JSONL format:

```python
# Input: blog_post.md
# Output: {"title": "...", "content": "...", "date": "..."}
```

### 2. Dataset Preparation (`src/dataset_prep.py`)
Transforms raw data into instruction-response pairs for fine-tuning:

```json
{
  "instruction": "Write a blog post about AI safety",
  "input": "",
  "output": "[Your actual blog content]"
}
```

### 3. Fine-tuning (`src/finetune.py`)
Uses **unsloth.ai** with **LoRA** (Low-Rank Adaptation) to efficiently fine-tune llama3.2:1b:
- LoRA adds small trainable matrices instead of updating all parameters
- Much faster and more memory-efficient than full fine-tuning
- Preserves base model knowledge while adapting to your style

### 4. Deployment (`src/deploy.py`)
Converts the fine-tuned model to Ollama format:
- Creates a Modelfile with your adapter
- Imports to local Ollama instance
- Makes it available via OpenAI-compatible API

### 5. Web Interface
- **Backend**: FastAPI serves both base and fine-tuned models
- **Frontend**: React UI for chatting with your AI twin
- **Features**: Model selection, side-by-side comparison, response streaming

## Key Technologies

| Technology | Purpose | Why We Use It |
|------------|---------|---------------|
| **llama3.2:1b** | Base LLM | Small enough to run locally, capable enough for text generation |
| **Ollama** | Model hosting | Easy local deployment, OpenAI-compatible API |
| **unsloth.ai** | Fine-tuning | 2x faster training, optimized for LoRA |
| **uv** | Package manager | 10-100x faster than pip, better dependency resolution |
| **pydantic-ai** | Model interface | Type-safe Python SDK, OpenAI compatibility |
| **FastAPI** | Backend API | Modern async Python framework, auto docs |
| **React** | Frontend UI | Component-based, responsive interface |
| **LoRA** | Fine-tuning method | Efficient parameter updates, preserves base knowledge |

## Evaluation and Comparison

The project includes tools to compare base vs. fine-tuned models:

```bash
# Run comprehensive evaluation
python cli.py evaluate --model blogging-twin:v1

# Output includes:
# - Perplexity scores
# - Style similarity metrics
# - Side-by-side sample generations
# - BLEU/ROUGE scores against original blogs
```

Example comparison:

```
Prompt: "Write about machine learning explainability"

Base Model:
"Machine learning explainability refers to understanding how models make decisions..."

Fine-tuned Model (Your Style):
"Let me tell you about ML explainability - it's fascinating stuff. When I first..."
```

## Customization Ideas

1. **Different base models**: Try llama3:8b or other models
2. **Multiple writing styles**: Fine-tune separate models for technical vs. casual writing
3. **Blog generation assistant**: Add structured prompts for different blog sections
4. **Continuous learning**: Periodically retrain on new blogs
5. **Style transfer controls**: Add sliders for formality, creativity, length

## Adding New Dependencies

This project uses `uv` for modern dependency management:

```bash
# Add a new package
uv add package-name

# Add a development dependency
uv add --dev pytest-cov

# Add a specific version
uv add "requests>=2.31.0"

# Update dependencies
uv sync

# Update uv.lock after manual pyproject.toml edits
uv lock
```

The `uv.lock` file ensures reproducible builds - commit it to git!

## Troubleshooting

### Ollama not found
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Start Ollama (if installed via package manager)
ollama serve
```

### Out of memory during fine-tuning
```yaml
# Reduce batch size in pipeline_config.yaml
training:
  batch_size: 2  # or 1
```

### Model not generating good results
- Ensure you have at least 10-20 blog posts for training
- Try increasing epochs to 5-10
- Adjust learning rate (try 1e-4 or 3e-4)
- Check that training loss is decreasing

## Learning Resources

### Included in this project:
- `REQUIREMENTS_AND_DESIGN.md` - Complete system design
- `CLAUDE.md` - Architecture and patterns
- Inline code comments explaining each step

### External resources:
- [uv Documentation](https://docs.astral.sh/uv/)
- [Ollama Documentation](https://ollama.ai/docs)
- [Unsloth GitHub](https://github.com/unslothai/unsloth)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [pydantic-ai Docs](https://ai.pydantic.dev/)

## Contributing

This is an educational project! Contributions are welcome:
- Bug fixes and improvements
- Additional data formats (PDF, HTML)
- New evaluation metrics
- UI enhancements
- Documentation improvements

## License

MIT License - Feel free to use this project for learning and building!

## Acknowledgments

- **Meta AI** for Llama models
- **Ollama** for local hosting tools
- **unsloth.ai** for optimized fine-tuning
- The open-source AI community

## Questions or Feedback?

- Open an issue for bugs or questions
- Star the repo if you found it helpful!
- Share your fine-tuned models and results

---

**Happy Learning and Building!** 🚀

*Remember: The goal isn't perfection, it's learning how these systems work. Experiment, break things, and understand the pieces.*
