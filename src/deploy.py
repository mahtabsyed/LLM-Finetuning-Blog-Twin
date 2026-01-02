"""
Deployment Module

This module handles deploying the fine-tuned model to a local Ollama instance.
It converts the fine-tuned LoRA weights into a format that Ollama can use.

Key concepts:
- Ollama uses Modelfiles (similar to Dockerfiles) to define models
- We need to merge LoRA adapters with the base model or reference them
- The deployed model becomes available via Ollama's API

Key functions:
- create_modelfile(): Generate Ollama Modelfile
- export_model_to_gguf(): Convert to Ollama-compatible format
- import_to_ollama(): Register model with Ollama
"""

import json
import logging
import subprocess
import requests
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)


def check_ollama_installed() -> bool:
    """
    Check if Ollama is installed and accessible.

    Returns:
        True if Ollama is installed, False otherwise
    """
    try:
        result = subprocess.run(
            ['ollama', '--version'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            logger.info(f"✓ Ollama is installed: {result.stdout.strip()}")
            return True
        else:
            logger.error("✗ Ollama command failed")
            return False
    except FileNotFoundError:
        logger.error("✗ Ollama not found. Please install from https://ollama.ai")
        return False
    except Exception as e:
        logger.error(f"✗ Error checking Ollama: {e}")
        return False


def check_ollama_running(base_url: str = "http://localhost:11434") -> bool:
    """
    Check if Ollama server is running.

    Args:
        base_url: Ollama API base URL

    Returns:
        True if server is running, False otherwise
    """
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=5)
        if response.status_code == 200:
            logger.info("✓ Ollama server is running")
            return True
        else:
            logger.warning(f"⚠ Ollama server returned status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        logger.error("✗ Cannot connect to Ollama server")
        logger.error("  Please start Ollama: ollama serve")
        return False
    except Exception as e:
        logger.error(f"✗ Error connecting to Ollama: {e}")
        return False


def check_base_model_available(model_name: str = "llama3.2:1b") -> bool:
    """
    Check if the base model is available in Ollama.

    Args:
        model_name: Name of the base model

    Returns:
        True if model is available, False otherwise
    """
    try:
        result = subprocess.run(
            ['ollama', 'list'],
            capture_output=True,
            text=True,
            timeout=10
        )

        if result.returncode == 0:
            # Check if model name appears in the output
            if model_name in result.stdout:
                logger.info(f"✓ Base model '{model_name}' is available")
                return True
            else:
                logger.warning(f"⚠ Base model '{model_name}' not found")
                logger.warning(f"  Please pull it: ollama pull {model_name}")
                return False
        else:
            logger.error("✗ Failed to list Ollama models")
            return False

    except Exception as e:
        logger.error(f"✗ Error checking base model: {e}")
        return False


def create_modelfile(
    base_model: str,
    adapter_path: Optional[str],
    model_name: str,
    parameters: Dict,
    output_dir: str
) -> str:
    """
    Create an Ollama Modelfile for the fine-tuned model.

    A Modelfile defines how Ollama should load and configure the model.

    Args:
        base_model: Name of the base model in Ollama
        adapter_path: Path to LoRA adapter (if separate)
        model_name: Name for the new model
        parameters: Model generation parameters
        output_dir: Directory to save Modelfile

    Returns:
        Path to created Modelfile
    """
    logger.info("Creating Ollama Modelfile...")

    # Build Modelfile content
    modelfile_content = f"""# Modelfile for {model_name}
# This model is a fine-tuned version of {base_model}
# Fine-tuned on personal blog posts

FROM {base_model}

"""

    # Add adapter if provided
    if adapter_path:
        modelfile_content += f"# Load LoRA adapter\nADAPTER {adapter_path}\n\n"

    # Add parameters
    modelfile_content += "# Generation parameters\n"
    for param, value in parameters.items():
        modelfile_content += f"PARAMETER {param} {value}\n"

    # Add system message (optional, but helps set context)
    modelfile_content += """
# System message to set the context
SYSTEM You are a helpful AI assistant that writes in a personal, engaging blog style. You have been trained on a collection of blog posts to understand the author's voice and perspective.
"""

    # Save Modelfile
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    modelfile_path = output_path / "Modelfile"

    with open(modelfile_path, 'w') as f:
        f.write(modelfile_content)

    logger.info(f"✓ Modelfile created: {modelfile_path}")

    return str(modelfile_path)


def merge_and_export_model(
    model_path: str,
    output_dir: str,
    base_model: str = "llama3.2:1b"
) -> str:
    """
    Merge LoRA adapters with base model and export.

    This function uses unsloth to merge the adapters and export to GGUF format
    (the format Ollama uses).

    Args:
        model_path: Path to fine-tuned model checkpoint
        output_dir: Directory to save merged model
        base_model: Base model name

    Returns:
        Path to exported model
    """
    logger.info("Merging LoRA adapters with base model...")
    logger.info("(This may take a few minutes)")

    try:
        from unsloth import FastLanguageModel

        # Load the fine-tuned model
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path,
            max_seq_length=2048,
            dtype=None,
            load_in_4bit=True,
        )

        # Merge LoRA weights with base model
        # This creates a single model without separate adapter files
        model = model.merge_and_unload()

        # Export to GGUF format for Ollama
        output_path = Path(output_dir) / "merged_model.gguf"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save in GGUF format with quantization
        # Q4_K_M is a good balance of quality and size
        model.save_pretrained_gguf(
            str(output_path.parent),
            tokenizer,
            quantization_method="q4_k_m"  # 4-bit quantization
        )

        logger.info(f"✓ Model merged and exported to {output_path}")
        return str(output_path)

    except Exception as e:
        logger.error(f"✗ Error merging model: {e}")
        logger.warning("Falling back to adapter-based deployment")
        return None


def import_to_ollama(
    modelfile_path: str,
    model_name: str
) -> bool:
    """
    Import the model into Ollama using the Modelfile.

    Args:
        modelfile_path: Path to Modelfile
        model_name: Name to register the model as

    Returns:
        True if successful, False otherwise
    """
    logger.info(f"Importing model to Ollama as '{model_name}'...")

    try:
        # Use 'ollama create' command to import the model
        result = subprocess.run(
            ['ollama', 'create', model_name, '-f', modelfile_path],
            capture_output=True,
            text=True,
            timeout=300  # 5 minutes timeout for large models
        )

        if result.returncode == 0:
            logger.info(f"✓ Model successfully imported as '{model_name}'")
            logger.info(f"  Test it: ollama run {model_name}")
            return True
        else:
            logger.error(f"✗ Failed to import model")
            logger.error(f"  Error: {result.stderr}")
            return False

    except subprocess.TimeoutExpired:
        logger.error("✗ Model import timed out (took > 5 minutes)")
        return False
    except Exception as e:
        logger.error(f"✗ Error importing model: {e}")
        return False


def verify_deployment(model_name: str, base_url: str = "http://localhost:11434") -> bool:
    """
    Verify that the deployed model is working.

    Tests the model with a simple prompt to ensure it's responding.

    Args:
        model_name: Name of the deployed model
        base_url: Ollama API base URL

    Returns:
        True if model responds correctly, False otherwise
    """
    logger.info(f"Verifying deployment of '{model_name}'...")

    try:
        # Send a simple test prompt
        response = requests.post(
            f"{base_url}/api/generate",
            json={
                "model": model_name,
                "prompt": "Hello, write a short sentence.",
                "stream": False
            },
            timeout=30
        )

        if response.status_code == 200:
            result = response.json()
            generated_text = result.get('response', '')
            logger.info(f"✓ Model is responding correctly")
            logger.info(f"  Test response: {generated_text[:100]}...")
            return True
        else:
            logger.error(f"✗ Model returned status {response.status_code}")
            return False

    except Exception as e:
        logger.error(f"✗ Error verifying deployment: {e}")
        return False


def run_deployment(
    model_path: str,
    model_name: str,
    config: Dict
) -> str:
    """
    Main entry point for the deployment module.

    This function orchestrates the complete deployment process:
    1. Verify Ollama is installed and running
    2. Check base model availability
    3. Create Modelfile
    4. (Optionally) Merge and export model
    5. Import to Ollama
    6. Verify deployment

    Args:
        model_path: Path to fine-tuned model checkpoint
        model_name: Name to deploy the model as in Ollama
        config: Configuration dictionary

    Returns:
        Deployed model name
    """
    logger.info("=" * 60)
    logger.info("Starting model deployment to Ollama")
    logger.info("=" * 60)
    logger.info(f"Model path: {model_path}")
    logger.info(f"Target name: {model_name}")
    logger.info("")

    # Extract configuration
    deploy_config = config.get('deployment', {})
    base_url = deploy_config.get('ollama_base_url', 'http://localhost:11434')
    parameters = deploy_config.get('parameters', {})
    base_model = config.get('training', {}).get('base_model', 'llama3.2:1b')

    # Step 1: Check Ollama installation
    if not check_ollama_installed():
        raise RuntimeError("Ollama is not installed. Please install from https://ollama.ai")

    # Step 2: Check Ollama server
    if not check_ollama_running(base_url):
        raise RuntimeError("Ollama server is not running. Please start it with: ollama serve")

    # Step 3: Check base model
    if not check_base_model_available(base_model):
        logger.warning(f"Base model not found. Attempting to pull...")
        try:
            subprocess.run(['ollama', 'pull', base_model], check=True)
        except:
            raise RuntimeError(f"Failed to pull base model: {base_model}")

    # Step 4: Create Modelfile
    # For simplicity, we'll use adapter-based deployment
    # In production, you might want to merge the model
    adapter_path = str(Path(model_path) / "adapter_model.safetensors")

    modelfile_path = create_modelfile(
        base_model=base_model,
        adapter_path=adapter_path if Path(adapter_path).exists() else None,
        model_name=model_name,
        parameters=parameters,
        output_dir=model_path
    )

    # Step 5: Import to Ollama
    success = import_to_ollama(modelfile_path, model_name)

    if not success:
        raise RuntimeError("Failed to import model to Ollama")

    # Step 6: Verify deployment
    if not verify_deployment(model_name, base_url):
        logger.warning("⚠ Model imported but verification failed")
        logger.warning("  The model may still work, but please test manually")

    logger.info("")
    logger.info("=" * 60)
    logger.info("Deployment complete!")
    logger.info("=" * 60)
    logger.info(f"Model '{model_name}' is ready to use")
    logger.info(f"Test it with: ollama run {model_name}")
    logger.info("")

    return model_name


if __name__ == '__main__':
    # Example usage when running this module directly
    import yaml

    # Load config
    with open('pipeline_config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Run deployment
    run_deployment(
        model_path='./models/checkpoints/blogging_twin',
        model_name='blogging-twin:latest',
        config=config
    )
