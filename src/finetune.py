"""
Fine-tuning Module

This module handles fine-tuning the llama3.2:1b model using unsloth.ai.
unsloth provides optimized implementations of LoRA (Low-Rank Adaptation) for efficient training.

Key concepts:
- LoRA: Adds small trainable matrices to model layers instead of training all parameters
- Efficient: Much faster and less memory-intensive than full fine-tuning
- Preserves: Keeps base model knowledge while adapting to your writing style

Key functions:
- load_base_model(): Load llama3.2:1b with LoRA adapters
- prepare_training_dataset(): Format data for the trainer
- train_model(): Execute the fine-tuning process
- save_model(): Save fine-tuned weights
"""

import json
import logging
from pathlib import Path
from typing import Dict, List
from datasets import Dataset
import torch

logger = logging.getLogger(__name__)


def check_gpu_availability():
    """
    Check if GPU is available for training.

    Returns information about GPU status for the user.
    """
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        logger.info(f"✓ GPU available: {gpu_name}")
        logger.info(f"  CUDA version: {torch.version.cuda}")
        return True
    else:
        logger.warning("⚠ No GPU detected - training will use CPU (slower)")
        logger.warning("  Consider using Google Colab for GPU access")
        return False


def load_base_model(
    model_name: str = "unsloth/llama-3.2-1b",
    max_seq_length: int = 2048,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    target_modules: List[str] = None
):
    """
    Load the base model with LoRA configuration.

    This function uses unsloth's FastLanguageModel to load the model
    with optimized LoRA adapters already attached.

    Args:
        model_name: HuggingFace model name (unsloth provides optimized versions)
        max_seq_length: Maximum sequence length for training
        lora_r: LoRA rank (higher = more parameters, better quality, slower)
        lora_alpha: LoRA scaling factor (typically 2x the rank)
        lora_dropout: Dropout probability for LoRA layers
        target_modules: Which model layers to apply LoRA to

    Returns:
        Tuple of (model, tokenizer)
    """
    from unsloth import FastLanguageModel

    logger.info("Loading base model with LoRA configuration...")
    logger.info(f"  Model: {model_name}")
    logger.info(f"  Max sequence length: {max_seq_length}")
    logger.info(f"  LoRA rank (r): {lora_r}")
    logger.info(f"  LoRA alpha: {lora_alpha}")

    # Default target modules for Llama models
    if target_modules is None:
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ]

    # Load model with 4-bit quantization for efficiency
    # This reduces memory usage significantly
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=None,  # Auto-detect best dtype
        load_in_4bit=True,  # Use 4-bit quantization
    )

    # Add LoRA adapters to the model
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias="none",
        use_gradient_checkpointing="unsloth",  # Use unsloth's optimized checkpointing
        random_state=42,
    )

    logger.info("✓ Model loaded with LoRA adapters")

    return model, tokenizer


def load_training_data(file_path: str) -> List[Dict]:
    """
    Load training data from JSONL file.

    Args:
        file_path: Path to training data JSONL

    Returns:
        List of training examples
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))

    logger.info(f"Loaded {len(data)} training examples")
    return data


def format_instruction_prompt(instruction: str, input_text: str, output: str = None) -> str:
    """
    Format a single example into the Alpaca instruction format.

    This is a popular format for instruction fine-tuning:
    ### Instruction: [task description]
    ### Input: [optional context]
    ### Response: [expected output]

    Args:
        instruction: The instruction/task
        input_text: Optional additional context
        output: The expected response (for training)

    Returns:
        Formatted prompt string
    """
    # Build the prompt
    prompt = f"### Instruction:\n{instruction}\n\n"

    if input_text:
        prompt += f"### Input:\n{input_text}\n\n"

    prompt += "### Response:\n"

    if output:
        prompt += output

    return prompt


def prepare_training_dataset(
    data: List[Dict],
    tokenizer,
    max_length: int = 2048
) -> Dataset:
    """
    Prepare the dataset for training.

    This function:
    1. Formats each example into the instruction prompt format
    2. Tokenizes the text
    3. Creates a HuggingFace Dataset object

    Args:
        data: List of training examples
        tokenizer: Model tokenizer
        max_length: Maximum sequence length

    Returns:
        HuggingFace Dataset ready for training
    """
    logger.info("Preparing training dataset...")

    # Format all examples
    formatted_examples = []
    for example in data:
        formatted_text = format_instruction_prompt(
            instruction=example['instruction'],
            input_text=example['input'],
            output=example['output']
        )
        formatted_examples.append({"text": formatted_text})

    # Create HuggingFace dataset
    dataset = Dataset.from_list(formatted_examples)

    logger.info(f"✓ Prepared {len(dataset)} examples for training")

    return dataset


def train_model(
    model,
    tokenizer,
    train_dataset: Dataset,
    output_dir: str,
    num_epochs: int = 3,
    learning_rate: float = 2e-4,
    batch_size: int = 4,
    gradient_accumulation_steps: int = 4,
    warmup_steps: int = 100,
    save_steps: int = 500,
    logging_steps: int = 10,
):
    """
    Train the model using the prepared dataset.

    This function uses HuggingFace's Trainer with unsloth optimizations.

    Args:
        model: The model with LoRA adapters
        tokenizer: Model tokenizer
        train_dataset: Prepared training dataset
        output_dir: Directory to save checkpoints
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        batch_size: Training batch size
        gradient_accumulation_steps: Steps to accumulate gradients
        warmup_steps: Number of warmup steps for learning rate
        save_steps: Save checkpoint every N steps
        logging_steps: Log metrics every N steps

    Returns:
        Trained model
    """
    from transformers import TrainingArguments
    from trl import SFTTrainer

    logger.info("=" * 60)
    logger.info("Starting model training")
    logger.info("=" * 60)
    logger.info(f"Epochs: {num_epochs}")
    logger.info(f"Learning rate: {learning_rate}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Gradient accumulation: {gradient_accumulation_steps}")
    logger.info(f"Effective batch size: {batch_size * gradient_accumulation_steps}")
    logger.info("")

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        fp16=not torch.cuda.is_bf16_supported(),  # Use fp16 unless bf16 is available
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=logging_steps,
        save_steps=save_steps,
        save_total_limit=3,  # Keep only last 3 checkpoints
        warmup_steps=warmup_steps,
        optim="adamw_8bit",  # Use 8-bit AdamW optimizer for efficiency
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=42,
        report_to="none",  # Disable wandb/tensorboard for simplicity
    )

    # Create SFT (Supervised Fine-Tuning) trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        dataset_text_field="text",  # Field containing the formatted text
        max_seq_length=2048,
        args=training_args,
        packing=False,  # Don't pack multiple examples per sequence
    )

    # Train the model
    logger.info("Starting training... (this may take a while)")
    logger.info("")

    trainer.train()

    logger.info("")
    logger.info("=" * 60)
    logger.info("Training complete!")
    logger.info("=" * 60)

    return model


def save_model(
    model,
    tokenizer,
    output_dir: str
):
    """
    Save the fine-tuned model and tokenizer.

    Saves:
    - LoRA adapter weights
    - Tokenizer files
    - Model configuration

    Args:
        model: Trained model
        tokenizer: Model tokenizer
        output_dir: Directory to save model files
    """
    logger.info(f"Saving model to {output_dir}...")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save LoRA adapters
    model.save_pretrained(output_path)

    # Save tokenizer
    tokenizer.save_pretrained(output_path)

    logger.info("✓ Model saved successfully")
    logger.info(f"  Location: {output_dir}")
    logger.info(f"  Files: adapter_config.json, adapter_model.safetensors, tokenizer files")


def run_finetuning(
    training_data_path: str,
    output_dir: str,
    config: Dict
) -> str:
    """
    Main entry point for the fine-tuning module.

    This function orchestrates the complete fine-tuning process:
    1. Check GPU availability
    2. Load base model with LoRA
    3. Load and prepare training data
    4. Train the model
    5. Save the fine-tuned model

    Args:
        training_data_path: Path to training data JSONL
        output_dir: Directory to save the fine-tuned model
        config: Configuration dictionary

    Returns:
        Path to the saved model
    """
    # Extract configuration
    train_config = config.get('training', {})
    lora_config = train_config.get('lora', {})

    # Check GPU
    check_gpu_availability()

    # Load base model with LoRA
    model, tokenizer = load_base_model(
        model_name="unsloth/llama-3.2-1b",
        max_seq_length=train_config.get('max_seq_length', 2048),
        lora_r=lora_config.get('r', 16),
        lora_alpha=lora_config.get('alpha', 32),
        lora_dropout=lora_config.get('dropout', 0.05),
        target_modules=lora_config.get('target_modules')
    )

    # Load training data
    training_data = load_training_data(training_data_path)

    # Prepare dataset
    train_dataset = prepare_training_dataset(
        data=training_data,
        tokenizer=tokenizer,
        max_length=train_config.get('max_seq_length', 2048)
    )

    # Train model
    trained_model = train_model(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        output_dir=output_dir,
        num_epochs=train_config.get('epochs', 3),
        learning_rate=train_config.get('learning_rate', 2e-4),
        batch_size=train_config.get('batch_size', 4),
        gradient_accumulation_steps=train_config.get('gradient_accumulation_steps', 4),
        warmup_steps=train_config.get('warmup_steps', 100),
        save_steps=train_config.get('save_steps', 500),
        logging_steps=train_config.get('logging_steps', 10),
    )

    # Save model
    save_model(trained_model, tokenizer, output_dir)

    return output_dir


if __name__ == '__main__':
    # Example usage when running this module directly
    import yaml

    # Load config
    with open('pipeline_config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Run fine-tuning
    run_finetuning(
        training_data_path='./data/processed/training_data.jsonl',
        output_dir='./models/checkpoints/blogging_twin',
        config=config
    )
