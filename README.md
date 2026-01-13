# LLM Blogging Twin

> Fine-tune llama3.2:1b on your blog posts to create an AI that writes like you.

![Project Type](https://img.shields.io/badge/Type-Educational-blue)
![Python](https://img.shields.io/badge/Python-3.10+-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

## What This Does

Fine-tunes a local LLM to replicate your writing style with:
- Modular CLI pipeline (run individual steps or complete workflow)
- Platform-aware training (Apple Silicon MLX / NVIDIA CUDA auto-detection)
- Local deployment with Ollama (no cloud required)
- Interactive web interface for text generation
- Side-by-side comparison of base vs fine-tuned models

**What you'll learn**: LLM fine-tuning with LoRA, local model deployment, FastAPI backends, React frontends, and MLOps pipeline design.

## Architecture

```
Blog Files → Data Ingestion → Dataset Prep → Fine-tuning → Ollama Deployment → Web Interface
```

Three-tier system: CLI Layer (orchestration) → Processing Layer (ML pipeline) → Interface Layer (API + UI)

See [DESIGN.md](DESIGN.md) for detailed architecture.

## Quick Start

### Prerequisites

| Requirement | Version |
|-------------|---------|
| Python | 3.10+ |
| Node.js | 18+ |
| RAM | 8GB min (16GB recommended) |
| Storage | 10GB free |
| GPU | Apple Silicon or NVIDIA recommended |

See [REQUIREMENTS.md](REQUIREMENTS.md) for detailed requirements.

### Installation

```bash
# 1. Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Clone repository
git clone https://github.com/yourusername/llm-blogging-twin.git
cd llm-blogging-twin

# 3. Install dependencies (auto-detects Apple Silicon/CUDA)
uv sync

# 4. Install Ollama and pull base model
curl https://ollama.ai/install.sh | sh
ollama pull llama3.2:1b

# 5. Setup frontend
cd frontend && npm install
```

### Run the Pipeline

Place your blog files (.md, .txt, or .docx) in `data/raw/`, then:

```bash
# Option 1: Full pipeline (recommended for first time)
uv run python cli.py pipeline --blog-dir ./data/raw --model-name blogging-twin:v1

# Option 2: Individual steps (for learning/debugging)
uv run python cli.py ingest --input-dir ./data/raw --output ./data/processed/raw_blogs.jsonl
uv run python cli.py prepare-dataset --input ./data/processed/raw_blogs.jsonl --output ./data/processed/training_data.jsonl
uv run python cli.py finetune --data ./data/processed/training_data.jsonl --output ./models/blogging_twin
uv run python cli.py deploy --model-path ./models/blogging_twin --model-name blogging-twin:v1
uv run python cli.py evaluate --model blogging-twin:v1
```

### Launch Web Interface

```bash
# Terminal 1: Backend
uv run python cli.py serve --port 8000

# Terminal 2: Frontend
cd frontend && npm run dev
```

Visit `http://localhost:5173` to interact with your AI twin.

## Platform Support

Automatic hardware detection during `uv sync`:

| Platform | Library | Framework |
|----------|---------|-----------|
| **Apple Silicon** (M1-M5) | unsloth-mlx | MLX |
| **NVIDIA GPU** | unsloth | CUDA |
| **AMD GPU** | unsloth | ROCm |
| **Intel GPU** | unsloth | XPU |

No manual configuration needed - the system detects your hardware and installs the correct dependencies.

## Project Structure

```
llm-blogging-twin/
├── README.md              # This file
├── REQUIREMENTS.md        # Dependencies & installation
├── DESIGN.md              # Architecture & components
├── CLAUDE.md              # Context for Claude Code
├── pyproject.toml         # Dependencies (source of truth)
├── uv.lock                # Locked dependencies
├── cli.py                 # CLI entry point
├── pipeline_config.yaml   # Configuration
│
├── src/                   # Processing modules
│   ├── data_ingestion.py
│   ├── dataset_prep.py
│   ├── finetune.py
│   ├── deploy.py
│   ├── evaluate.py
│   └── pipeline.py
│
├── api/                   # FastAPI backend
├── frontend/              # React web interface
├── data/                  # Data storage
│   ├── raw/               # Your blog files go here
│   ├── processed/         # Generated datasets
│   └── validation/        # Test sets
├── models/                # Model checkpoints
└── results/               # Evaluation reports
```

## Key Commands

| Command | Purpose | Example |
|---------|---------|---------|
| `pipeline` | Run complete workflow | `uv run python cli.py pipeline --blog-dir ./data/raw` |
| `ingest` | Parse blog files | `uv run python cli.py ingest --input-dir ./data/raw` |
| `prepare-dataset` | Format training data | `uv run python cli.py prepare-dataset --input raw.jsonl` |
| `finetune` | Train model | `uv run python cli.py finetune --data training.jsonl` |
| `deploy` | Deploy to Ollama | `uv run python cli.py deploy --model-path ./models/v1` |
| `evaluate` | Compare models | `uv run python cli.py evaluate --model blogging-twin:v1` |
| `serve` | Start API server | `uv run python cli.py serve --port 8000` |

## Configuration

Edit `pipeline_config.yaml` to customize training:

```yaml
data:
  input_dir: ./data/raw
  formats: [md, txt, docx]
  train_split: 0.8

training:
  epochs: 3
  learning_rate: 2e-4
  batch_size: 4
  lora_r: 16           # LoRA rank
  lora_alpha: 32       # LoRA alpha

deployment:
  model_name: blogging-twin:latest
  ollama_base_url: http://localhost:11434

evaluation:
  run_after_training: true
```

## How It Works

### 1. Data Ingestion
Reads blog files and converts to JSONL format with metadata (title, content, date, word count).

### 2. Dataset Preparation
Transforms blogs into instruction-response pairs for fine-tuning:
```json
{
  "instruction": "Write a blog post about AI safety",
  "input": "",
  "output": "[Your actual blog content]"
}
```

### 3. Fine-tuning
Uses unsloth with LoRA (Low-Rank Adaptation) for efficient training. LoRA adds small trainable matrices instead of updating all parameters, making it faster and more memory-efficient while preserving base knowledge.

### 4. Deployment
Converts fine-tuned model to Ollama format and imports to local instance, making it available via OpenAI-compatible API.

### 5. Web Interface
FastAPI backend serves both models (base and fine-tuned). React frontend provides chat interface with model selection and side-by-side comparison.

## Key Technologies

| Technology | Purpose |
|------------|---------|
| **llama3.2:1b** | Base model - small enough for local training |
| **Ollama** | Local hosting with OpenAI-compatible API |
| **unsloth/unsloth-mlx** | Optimized fine-tuning (platform-specific) |
| **uv** | Fast Python dependency management |
| **pydantic-ai** | Type-safe model interface |
| **FastAPI** | Async Python web framework |
| **React** | Interactive UI components |
| **LoRA** | Efficient fine-tuning method |

## Python API Usage

```python
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

# Connect to fine-tuned model
model = OpenAIModel(
    model_name="blogging-twin:latest",
    provider=OpenAIProvider(base_url="http://localhost:11434/v1")
)

# Generate content
response = model.generate("Write about the future of AI")
print(response)
```

## Customization Ideas

1. **Different base models**: Try llama3:8b or other Ollama-compatible models
2. **Multiple styles**: Fine-tune separate models for technical vs casual writing
3. **Continuous learning**: Periodically retrain on new blog posts
4. **Style controls**: Add sliders for formality, creativity, length in UI
5. **Blog assistant**: Structured prompts for intros, conclusions, outlines

## Dependency Management

This project uses **uv** for modern Python dependency management:

```bash
# Add new package
uv add package-name

# Add dev dependency
uv add --dev pytest-cov

# Update dependencies
uv sync
```

**Important**: `pyproject.toml` is the source of truth, not `requirements.txt`. The `uv.lock` file ensures reproducible builds - commit it to git.

## Troubleshooting

### Ollama not found
```bash
# Check if running
curl http://localhost:11434

# Start Ollama
ollama serve
```

### Out of memory during fine-tuning
Reduce batch size in `pipeline_config.yaml`:
```yaml
training:
  batch_size: 2  # or 1
```

### Model not generating good results
- Ensure 10-20+ blog posts for training
- Increase epochs to 5-10
- Adjust learning rate (try 1e-4 or 3e-4)
- Check training loss is decreasing

### Platform detection issues
```bash
# Check architecture
uname -m

# For Apple Silicon, ensure native Python (not Rosetta)
python -c "import platform; print(platform.machine())"
```

## Evaluation

Compare base vs fine-tuned models:

```bash
uv run python cli.py evaluate --model blogging-twin:v1
```

Output includes:
- Perplexity scores
- Style similarity metrics
- Side-by-side sample generations
- BLEU/ROUGE scores

Example comparison:

**Prompt**: "Write about machine learning explainability"

**Base Model**: "Machine learning explainability refers to understanding how models make decisions..."

**Fine-tuned Model**: "Let me tell you about ML explainability - it's fascinating stuff. When I first..."

## Documentation

- **README.md** (this file): Quick start and overview
- **REQUIREMENTS.md**: Detailed dependencies and installation
- **DESIGN.md**: System architecture and component design
- **CLAUDE.md**: Context for Claude Code (AI assistant)

## Resources

- [uv Documentation](https://docs.astral.sh/uv/)
- [Ollama Documentation](https://ollama.ai/docs)
- [Unsloth GitHub](https://github.com/unslothai/unsloth)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [pydantic-ai Docs](https://ai.pydantic.dev/)

## Contributing

Contributions welcome:
- Bug fixes and improvements
- Additional data formats (PDF, HTML)
- New evaluation metrics
- UI enhancements
- Documentation improvements

## License

MIT License - Feel free to use for learning and building.

## Acknowledgments

- **Meta AI** for Llama models
- **Ollama** for local hosting tools
- **unsloth.ai** for optimized fine-tuning
- The open-source AI community

---

**Questions?** Open an issue or star the repo if you found it helpful!

*The goal is learning how these systems work. Experiment, break things, and understand the pieces.*
