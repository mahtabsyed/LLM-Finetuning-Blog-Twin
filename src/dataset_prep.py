"""
Dataset Preparation Module

This module converts raw blog data into instruction-response pairs suitable for fine-tuning.
It creates a dataset in the format expected by unsloth.ai and other fine-tuning frameworks.

Key functions:
- create_instruction_pairs(): Convert blogs to training examples
- split_dataset(): Create train/validation splits
- format_for_finetuning(): Format in the standard instruction format
"""

import json
import logging
import random
from pathlib import Path
from typing import List, Dict, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class InstructionPair:
    """
    Represents a single training example in instruction-response format.

    This format is commonly used for fine-tuning language models:
    - instruction: What the model should do
    - input: Optional context or additional information
    - output: The expected response (your blog content)
    """
    instruction: str
    input: str
    output: str


def load_jsonl(file_path: str) -> List[Dict]:
    """
    Load data from a JSONL file.

    Args:
        file_path: Path to JSONL file

    Returns:
        List of dictionaries, one per line
    """
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():  # Skip empty lines
                    data.append(json.loads(line))
        logger.info(f"Loaded {len(data)} entries from {file_path}")
        return data
    except Exception as e:
        logger.error(f"Error loading JSONL file {file_path}: {e}")
        raise


def generate_instruction_prompts(blog: Dict) -> List[str]:
    """
    Generate varied instruction prompts for a blog.

    This creates diversity in how the model learns to respond.
    Instead of always seeing "Write a blog post", it sees various phrasings.

    Args:
        blog: Blog dictionary with title and content

    Returns:
        List of possible instruction prompts
    """
    title = blog.get('title', 'a topic')

    # Create multiple instruction variations to increase training diversity
    prompts = [
        f"Write a blog post about {title}",
        f"Create an article discussing {title}",
        f"Write about {title} in your own style",
        f"Compose a blog post on the topic of {title}",
        f"Share your thoughts on {title}",
        f"Write an engaging post about {title}",
    ]

    return prompts


def create_instruction_pairs(blogs: List[Dict]) -> List[InstructionPair]:
    """
    Convert blog entries into instruction-response training pairs.

    Each blog can generate multiple training examples with different instructions.
    This increases dataset size and helps the model generalize better.

    Args:
        blogs: List of blog dictionaries from ingestion

    Returns:
        List of InstructionPair objects
    """
    pairs = []

    for blog in blogs:
        title = blog.get('title', 'Unknown')
        content = blog.get('content', '')

        if not content:
            logger.warning(f"Skipping blog '{title}': no content")
            continue

        # Generate instruction prompts for this blog
        instructions = generate_instruction_prompts(blog)

        # For each blog, we can create multiple training examples
        # with different instruction phrasings
        # Here we'll use the first instruction, but you could use all for more data
        instruction = random.choice(instructions)

        # Create the instruction pair
        pair = InstructionPair(
            instruction=instruction,
            input="",  # We don't use additional input for blog generation
            output=content
        )

        pairs.append(pair)

    logger.info(f"Created {len(pairs)} instruction-response pairs")
    return pairs


def split_dataset(
    pairs: List[InstructionPair],
    train_ratio: float = 0.8,
    shuffle: bool = True,
    random_seed: int = 42
) -> Tuple[List[InstructionPair], List[InstructionPair]]:
    """
    Split dataset into training and validation sets.

    Args:
        pairs: List of instruction pairs
        train_ratio: Fraction of data to use for training (0.8 = 80%)
        shuffle: Whether to shuffle data before splitting
        random_seed: Random seed for reproducibility

    Returns:
        Tuple of (train_pairs, validation_pairs)
    """
    # Set random seed for reproducible splits
    if shuffle:
        random.seed(random_seed)
        pairs_copy = pairs.copy()
        random.shuffle(pairs_copy)
    else:
        pairs_copy = pairs

    # Calculate split point
    split_idx = int(len(pairs_copy) * train_ratio)

    # Split the data
    train_pairs = pairs_copy[:split_idx]
    val_pairs = pairs_copy[split_idx:]

    logger.info(f"Dataset split: {len(train_pairs)} train, {len(val_pairs)} validation")

    return train_pairs, val_pairs


def format_for_finetuning(pairs: List[InstructionPair]) -> List[Dict]:
    """
    Format instruction pairs into the standard fine-tuning format.

    This creates the format expected by most fine-tuning libraries:
    {
        "instruction": "...",
        "input": "...",
        "output": "..."
    }

    Args:
        pairs: List of InstructionPair objects

    Returns:
        List of dictionaries in training format
    """
    formatted = []

    for pair in pairs:
        formatted.append({
            'instruction': pair.instruction,
            'input': pair.input,
            'output': pair.output
        })

    return formatted


def save_training_data(
    train_data: List[Dict],
    val_data: List[Dict],
    output_path: str,
    validation_dir: str = './data/validation'
) -> None:
    """
    Save training and validation data to JSONL files.

    Args:
        train_data: Training examples
        val_data: Validation examples
        output_path: Path for training data file
        validation_dir: Directory for validation data
    """
    # Save training data
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        for example in train_data:
            json_line = json.dumps(example, ensure_ascii=False)
            f.write(json_line + '\n')

    logger.info(f"✓ Saved {len(train_data)} training examples to {output_path}")

    # Save validation data
    val_dir = Path(validation_dir)
    val_dir.mkdir(parents=True, exist_ok=True)
    val_path = val_dir / 'validation.jsonl'

    with open(val_path, 'w', encoding='utf-8') as f:
        for example in val_data:
            json_line = json.dumps(example, ensure_ascii=False)
            f.write(json_line + '\n')

    logger.info(f"✓ Saved {len(val_data)} validation examples to {val_path}")


def analyze_dataset(data: List[Dict]) -> Dict:
    """
    Analyze dataset statistics for logging and verification.

    Args:
        data: List of training examples

    Returns:
        Dictionary with dataset statistics
    """
    if not data:
        return {}

    # Calculate statistics
    total_samples = len(data)
    output_lengths = [len(item['output'].split()) for item in data]

    stats = {
        'total_samples': total_samples,
        'avg_output_length': sum(output_lengths) / len(output_lengths),
        'min_output_length': min(output_lengths),
        'max_output_length': max(output_lengths),
    }

    return stats


def run_dataset_preparation(
    input_path: str,
    output_path: str,
    split_ratio: float,
    config: Dict
) -> Tuple[int, int]:
    """
    Main entry point for dataset preparation.

    This function orchestrates the complete dataset preparation process:
    1. Load raw blog data from JSONL
    2. Create instruction-response pairs
    3. Split into train/validation sets
    4. Format for fine-tuning
    5. Save to output files

    Args:
        input_path: Path to raw blogs JSONL file
        output_path: Path for training data output
        split_ratio: Train/validation split ratio
        config: Configuration dictionary

    Returns:
        Tuple of (train_size, validation_size)
    """
    logger.info("=" * 60)
    logger.info("Starting dataset preparation")
    logger.info("=" * 60)
    logger.info(f"Input: {input_path}")
    logger.info(f"Output: {output_path}")
    logger.info(f"Train/Val split: {split_ratio:.1%} / {(1-split_ratio):.1%}")
    logger.info("")

    # Load raw blog data
    blogs = load_jsonl(input_path)

    if not blogs:
        raise ValueError("No blogs found in input file!")

    # Create instruction-response pairs
    pairs = create_instruction_pairs(blogs)

    if not pairs:
        raise ValueError("No valid instruction pairs created!")

    # Split dataset
    train_pairs, val_pairs = split_dataset(
        pairs=pairs,
        train_ratio=split_ratio,
        shuffle=True
    )

    # Format for fine-tuning
    train_data = format_for_finetuning(train_pairs)
    val_data = format_for_finetuning(val_pairs)

    # Log dataset statistics
    logger.info("Training set statistics:")
    train_stats = analyze_dataset(train_data)
    for key, value in train_stats.items():
        if isinstance(value, float):
            logger.info(f"  {key}: {value:.2f}")
        else:
            logger.info(f"  {key}: {value}")

    logger.info("")
    logger.info("Validation set statistics:")
    val_stats = analyze_dataset(val_data)
    for key, value in val_stats.items():
        if isinstance(value, float):
            logger.info(f"  {key}: {value:.2f}")
        else:
            logger.info(f"  {key}: {value}")

    # Save datasets
    validation_dir = config.get('data', {}).get('validation_dir', './data/validation')
    save_training_data(
        train_data=train_data,
        val_data=val_data,
        output_path=output_path,
        validation_dir=validation_dir
    )

    logger.info("")
    logger.info("=" * 60)
    logger.info("Dataset preparation complete")
    logger.info("=" * 60)

    return len(train_data), len(val_data)


if __name__ == '__main__':
    # Example usage when running this module directly
    import yaml

    # Load config
    with open('pipeline_config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Run dataset preparation
    run_dataset_preparation(
        input_path='./data/processed/raw_blogs.jsonl',
        output_path='./data/processed/training_data.jsonl',
        split_ratio=0.8,
        config=config
    )
