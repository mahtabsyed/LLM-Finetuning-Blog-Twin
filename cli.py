#!/usr/bin/env python3
"""
LLM Blogging Twin - Command Line Interface

This CLI provides commands to run each stage of the blogging twin pipeline:
- ingest: Read and parse blog files
- prepare-dataset: Format data for fine-tuning
- finetune: Train the model using unsloth.ai
- deploy: Deploy model to Ollama
- evaluate: Assess model performance
- serve: Start the API server
- pipeline: Run the complete workflow

Example usage:
    python cli.py ingest --input-dir ./data/raw
    python cli.py pipeline --blog-dir ./data/raw --model-name blogging-twin:v1
"""

import click
import yaml
import logging
from pathlib import Path
from rich.console import Console
from rich.logging import RichHandler

# Initialize rich console for beautiful output
console = Console()

# Configure logging
logging.basicConfig(
    level="INFO",
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True, console=console)]
)
logger = logging.getLogger(__name__)


def load_config(config_path: str = "pipeline_config.yaml") -> dict:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to the YAML configuration file

    Returns:
        Dictionary containing configuration parameters
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"✓ Loaded configuration from {config_path}")
        return config
    except FileNotFoundError:
        logger.error(f"✗ Configuration file not found: {config_path}")
        raise
    except yaml.YAMLError as e:
        logger.error(f"✗ Error parsing YAML configuration: {e}")
        raise


@click.group()
@click.version_option(version="1.0.0", prog_name="LLM Blogging Twin")
def cli():
    """
    LLM Blogging Twin - Fine-tune a local LLM on your blog posts.

    This tool helps you create a personalized AI that writes in your style.
    """
    pass


@cli.command()
@click.option(
    '--input-dir',
    type=click.Path(exists=True),
    required=True,
    help='Directory containing blog files (.md, .txt, .docx)'
)
@click.option(
    '--output',
    type=click.Path(),
    default='./data/processed/raw_blogs.jsonl',
    help='Output JSONL file path'
)
@click.option(
    '--config',
    type=click.Path(exists=True),
    default='pipeline_config.yaml',
    help='Path to configuration file'
)
def ingest(input_dir: str, output: str, config: str):
    """
    Ingest blog files from a directory.

    Reads markdown (.md), text (.txt), and Word (.docx) files,
    extracting their content and metadata into a structured JSONL format.

    Example:
        python cli.py ingest --input-dir ./data/raw --output ./data/raw_blogs.jsonl
    """
    from src.data_ingestion import run_ingestion

    console.print("\n[bold cyan]📚 Starting Blog Ingestion[/bold cyan]\n")

    # Load configuration
    cfg = load_config(config)

    # Run ingestion
    try:
        num_blogs = run_ingestion(
            input_dir=input_dir,
            output_path=output,
            config=cfg
        )
        console.print(f"\n[bold green]✓ Successfully ingested {num_blogs} blog posts![/bold green]\n")
    except Exception as e:
        console.print(f"\n[bold red]✗ Ingestion failed: {e}[/bold red]\n")
        raise


@cli.command()
@click.option(
    '--input',
    type=click.Path(exists=True),
    required=True,
    help='Input JSONL file from ingestion'
)
@click.option(
    '--output',
    type=click.Path(),
    default='./data/processed/training_data.jsonl',
    help='Output training data file'
)
@click.option(
    '--split',
    type=float,
    default=0.8,
    help='Train/validation split ratio (default: 0.8)'
)
@click.option(
    '--config',
    type=click.Path(exists=True),
    default='pipeline_config.yaml',
    help='Path to configuration file'
)
def prepare_dataset(input: str, output: str, split: float, config: str):
    """
    Prepare training dataset from ingested blogs.

    Transforms raw blog data into instruction-response pairs suitable
    for fine-tuning. Splits data into training and validation sets.

    Example:
        python cli.py prepare-dataset --input ./data/raw_blogs.jsonl --split 0.8
    """
    from src.dataset_prep import run_dataset_preparation

    console.print("\n[bold cyan]🔧 Preparing Training Dataset[/bold cyan]\n")

    # Load configuration
    cfg = load_config(config)

    # Run dataset preparation
    try:
        train_size, val_size = run_dataset_preparation(
            input_path=input,
            output_path=output,
            split_ratio=split,
            config=cfg
        )
        console.print(f"\n[bold green]✓ Dataset prepared![/bold green]")
        console.print(f"  Training samples: {train_size}")
        console.print(f"  Validation samples: {val_size}\n")
    except Exception as e:
        console.print(f"\n[bold red]✗ Dataset preparation failed: {e}[/bold red]\n")
        raise


@cli.command()
@click.option(
    '--data',
    type=click.Path(exists=True),
    required=True,
    help='Training data JSONL file'
)
@click.option(
    '--output',
    type=click.Path(),
    default='./models/checkpoints/blogging_twin',
    help='Output directory for model checkpoints'
)
@click.option(
    '--epochs',
    type=int,
    default=3,
    help='Number of training epochs'
)
@click.option(
    '--lr',
    type=float,
    default=2e-4,
    help='Learning rate'
)
@click.option(
    '--config',
    type=click.Path(exists=True),
    default='pipeline_config.yaml',
    help='Path to configuration file'
)
def finetune(data: str, output: str, epochs: int, lr: float, config: str):
    """
    Fine-tune llama3.2:1b using unsloth.ai.

    Applies LoRA (Low-Rank Adaptation) to efficiently train the model
    on your blog posts while preserving base model knowledge.

    Example:
        python cli.py finetune --data ./data/training_data.jsonl --epochs 3
    """
    from src.finetune import run_finetuning

    console.print("\n[bold cyan]🚀 Starting Model Fine-tuning[/bold cyan]\n")
    console.print(f"  Epochs: {epochs}")
    console.print(f"  Learning rate: {lr}")
    console.print(f"  Output: {output}\n")

    # Load configuration
    cfg = load_config(config)

    # Override config with CLI arguments if provided
    if epochs:
        cfg['training']['epochs'] = epochs
    if lr:
        cfg['training']['learning_rate'] = lr

    # Run fine-tuning
    try:
        model_path = run_finetuning(
            training_data_path=data,
            output_dir=output,
            config=cfg
        )
        console.print(f"\n[bold green]✓ Fine-tuning complete![/bold green]")
        console.print(f"  Model saved to: {model_path}\n")
    except Exception as e:
        console.print(f"\n[bold red]✗ Fine-tuning failed: {e}[/bold red]\n")
        raise


@cli.command()
@click.option(
    '--model-path',
    type=click.Path(exists=True),
    required=True,
    help='Path to fine-tuned model checkpoint'
)
@click.option(
    '--model-name',
    type=str,
    default='blogging-twin:latest',
    help='Name for the model in Ollama (e.g., blogging-twin:v1)'
)
@click.option(
    '--config',
    type=click.Path(exists=True),
    default='pipeline_config.yaml',
    help='Path to configuration file'
)
def deploy(model_path: str, model_name: str, config: str):
    """
    Deploy fine-tuned model to local Ollama instance.

    Converts the fine-tuned model to Ollama format and makes it
    available for inference via the Ollama API.

    Example:
        python cli.py deploy --model-path ./models/blogging_twin --model-name blogging-twin:v1
    """
    from src.deploy import run_deployment

    console.print("\n[bold cyan]📦 Deploying Model to Ollama[/bold cyan]\n")
    console.print(f"  Model path: {model_path}")
    console.print(f"  Model name: {model_name}\n")

    # Load configuration
    cfg = load_config(config)

    # Run deployment
    try:
        deployed_name = run_deployment(
            model_path=model_path,
            model_name=model_name,
            config=cfg
        )
        console.print(f"\n[bold green]✓ Model deployed successfully![/bold green]")
        console.print(f"  Test it: ollama run {deployed_name}\n")
    except Exception as e:
        console.print(f"\n[bold red]✗ Deployment failed: {e}[/bold red]\n")
        raise


@cli.command()
@click.option(
    '--model',
    type=str,
    default='blogging-twin:latest',
    help='Model name in Ollama to evaluate'
)
@click.option(
    '--test-set',
    type=click.Path(exists=True),
    help='Optional test dataset (uses validation set if not provided)'
)
@click.option(
    '--output',
    type=click.Path(),
    default='./results/evaluations/eval_report.json',
    help='Output path for evaluation report'
)
@click.option(
    '--config',
    type=click.Path(exists=True),
    default='pipeline_config.yaml',
    help='Path to configuration file'
)
def evaluate(model: str, test_set: str, output: str, config: str):
    """
    Evaluate fine-tuned model performance.

    Compares base and fine-tuned models using various metrics:
    - Perplexity (model confidence)
    - Style similarity (writing pattern match)
    - BLEU/ROUGE scores (content overlap)

    Example:
        python cli.py evaluate --model blogging-twin:v1
    """
    from src.evaluate import run_evaluation

    console.print("\n[bold cyan]📊 Evaluating Model Performance[/bold cyan]\n")
    console.print(f"  Model: {model}\n")

    # Load configuration
    cfg = load_config(config)

    # Run evaluation
    try:
        report = run_evaluation(
            model_name=model,
            test_set_path=test_set,
            output_path=output,
            config=cfg
        )
        console.print(f"\n[bold green]✓ Evaluation complete![/bold green]")
        console.print(f"  Report saved to: {output}\n")
    except Exception as e:
        console.print(f"\n[bold red]✗ Evaluation failed: {e}[/bold red]\n")
        raise


@cli.command()
@click.option(
    '--port',
    type=int,
    default=8000,
    help='Port to run the API server on'
)
@click.option(
    '--host',
    type=str,
    default='0.0.0.0',
    help='Host to bind the server to'
)
@click.option(
    '--config',
    type=click.Path(exists=True),
    default='pipeline_config.yaml',
    help='Path to configuration file'
)
def serve(port: int, host: str, config: str):
    """
    Start the FastAPI backend server.

    Launches the API server that connects the React frontend
    to your local Ollama models (base and fine-tuned).

    Example:
        python cli.py serve --port 8000
    """
    import uvicorn

    console.print("\n[bold cyan]🌐 Starting API Server[/bold cyan]\n")
    console.print(f"  Host: {host}")
    console.print(f"  Port: {port}")
    console.print(f"  API docs: http://localhost:{port}/docs\n")
    console.print("[yellow]Press Ctrl+C to stop the server[/yellow]\n")

    # Load configuration
    cfg = load_config(config)

    # Start server
    try:
        uvicorn.run(
            "api.server:app",
            host=host,
            port=port,
            reload=True,  # Auto-reload on code changes (development mode)
            log_level="info"
        )
    except KeyboardInterrupt:
        console.print("\n[yellow]Server stopped by user[/yellow]\n")
    except Exception as e:
        console.print(f"\n[bold red]✗ Server failed: {e}[/bold red]\n")
        raise


@cli.command()
@click.option(
    '--blog-dir',
    type=click.Path(exists=True),
    required=True,
    help='Directory containing blog files'
)
@click.option(
    '--model-name',
    type=str,
    default='blogging-twin:latest',
    help='Name for the deployed model'
)
@click.option(
    '--config',
    type=click.Path(exists=True),
    default='pipeline_config.yaml',
    help='Path to configuration file'
)
@click.option(
    '--skip-evaluation',
    is_flag=True,
    help='Skip evaluation step'
)
def pipeline(blog_dir: str, model_name: str, config: str, skip_evaluation: bool):
    """
    Run the complete fine-tuning pipeline.

    Executes all steps in sequence:
    1. Ingest blog files
    2. Prepare training dataset
    3. Fine-tune model
    4. Deploy to Ollama
    5. Evaluate performance (optional)

    Example:
        python cli.py pipeline --blog-dir ./data/raw --model-name blogging-twin:v1
    """
    from src.pipeline import run_full_pipeline

    console.print("\n[bold magenta]🔄 Starting Full Pipeline[/bold magenta]\n")
    console.print(f"  Blog directory: {blog_dir}")
    console.print(f"  Model name: {model_name}")
    console.print(f"  Skip evaluation: {skip_evaluation}\n")

    # Load configuration
    cfg = load_config(config)

    # Run complete pipeline
    try:
        results = run_full_pipeline(
            blog_dir=blog_dir,
            model_name=model_name,
            config=cfg,
            skip_evaluation=skip_evaluation
        )
        console.print("\n[bold green]✓ Pipeline completed successfully![/bold green]\n")
        console.print(f"  Model deployed as: {model_name}")
        console.print(f"  Test it: ollama run {model_name}\n")
    except Exception as e:
        console.print(f"\n[bold red]✗ Pipeline failed: {e}[/bold red]\n")
        raise


if __name__ == '__main__':
    cli()
