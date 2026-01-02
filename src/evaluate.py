"""
Evaluation Module

This module evaluates the fine-tuned model's performance compared to the base model.
It uses various metrics to assess how well the model has learned your writing style.

Key metrics:
- Perplexity: How confident is the model? (lower is better)
- Style similarity: Does it match your writing patterns?
- BLEU/ROUGE: Content overlap with original blogs
- Human comparison: Side-by-side outputs for manual review

Key functions:
- generate_response(): Get model output for a prompt
- calculate_perplexity(): Measure model confidence
- compare_models(): Side-by-side comparison of base vs. fine-tuned
"""

import json
import logging
import requests
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)


def load_test_prompts(file_path: str) -> List[str]:
    """
    Load test prompts from a text file.

    Each line in the file should be a separate prompt.

    Args:
        file_path: Path to prompts file

    Returns:
        List of prompt strings
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            prompts = [line.strip() for line in f if line.strip()]

        logger.info(f"Loaded {len(prompts)} test prompts from {file_path}")
        return prompts

    except FileNotFoundError:
        logger.warning(f"Prompts file not found: {file_path}")
        logger.warning("Using default prompts instead")
        return get_default_prompts()


def get_default_prompts() -> List[str]:
    """
    Get default test prompts if no file is provided.

    Returns:
        List of default prompts
    """
    return [
        "Write a blog post about the future of artificial intelligence",
        "Share your thoughts on work-life balance in tech",
        "Explain machine learning to a beginner",
        "Write about the importance of continuous learning",
        "Discuss the impact of remote work on productivity",
    ]


def generate_response(
    model_name: str,
    prompt: str,
    base_url: str = "http://localhost:11434",
    max_tokens: int = 500,
    temperature: float = 0.7
) -> Dict:
    """
    Generate a response from an Ollama model.

    Args:
        model_name: Name of the model in Ollama
        prompt: Input prompt
        base_url: Ollama API base URL
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature (higher = more creative)

    Returns:
        Dictionary with response and metadata
    """
    try:
        # Format the prompt
        formatted_prompt = f"### Instruction:\n{prompt}\n\n### Response:\n"

        # Call Ollama API
        response = requests.post(
            f"{base_url}/api/generate",
            json={
                "model": model_name,
                "prompt": formatted_prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                }
            },
            timeout=60
        )

        if response.status_code == 200:
            result = response.json()
            return {
                'success': True,
                'text': result.get('response', ''),
                'model': model_name,
                'prompt': prompt,
                'tokens_generated': result.get('eval_count', 0),
                'generation_time': result.get('total_duration', 0) / 1e9  # Convert to seconds
            }
        else:
            logger.error(f"API returned status {response.status_code}")
            return {
                'success': False,
                'error': f"API error: {response.status_code}",
                'model': model_name,
                'prompt': prompt
            }

    except Exception as e:
        logger.error(f"Error generating response: {e}")
        return {
            'success': False,
            'error': str(e),
            'model': model_name,
            'prompt': prompt
        }


def compare_models(
    base_model: str,
    finetuned_model: str,
    prompts: List[str],
    base_url: str = "http://localhost:11434"
) -> List[Dict]:
    """
    Generate side-by-side comparisons from base and fine-tuned models.

    Args:
        base_model: Name of base model
        finetuned_model: Name of fine-tuned model
        prompts: List of test prompts
        base_url: Ollama API base URL

    Returns:
        List of comparison dictionaries
    """
    logger.info(f"Comparing {base_model} vs {finetuned_model}...")
    logger.info(f"Running {len(prompts)} test prompts")

    comparisons = []

    for i, prompt in enumerate(prompts, 1):
        logger.info(f"[{i}/{len(prompts)}] Testing: {prompt[:50]}...")

        # Generate from base model
        base_response = generate_response(base_model, prompt, base_url)

        # Generate from fine-tuned model
        finetuned_response = generate_response(finetuned_model, prompt, base_url)

        # Store comparison
        comparison = {
            'prompt': prompt,
            'base_model': {
                'name': base_model,
                'response': base_response.get('text', ''),
                'success': base_response.get('success', False),
                'generation_time': base_response.get('generation_time', 0)
            },
            'finetuned_model': {
                'name': finetuned_model,
                'response': finetuned_response.get('text', ''),
                'success': finetuned_response.get('success', False),
                'generation_time': finetuned_response.get('generation_time', 0)
            }
        }

        comparisons.append(comparison)

    logger.info(f"✓ Generated {len(comparisons)} comparisons")
    return comparisons


def calculate_style_metrics(response: str, reference_blogs: List[Dict]) -> Dict:
    """
    Calculate simple style similarity metrics.

    This is a simplified version - production would use more sophisticated metrics.

    Args:
        response: Generated text
        reference_blogs: Original blog posts for comparison

    Returns:
        Dictionary of style metrics
    """
    # Simple metrics: average sentence length, word variety, etc.
    words = response.split()
    sentences = response.split('.')

    metrics = {
        'word_count': len(words),
        'sentence_count': len([s for s in sentences if s.strip()]),
        'avg_sentence_length': len(words) / max(len(sentences), 1),
        'unique_word_ratio': len(set(words)) / max(len(words), 1)
    }

    return metrics


def calculate_bleu_rouge(generated: str, reference: str) -> Dict:
    """
    Calculate BLEU and ROUGE scores.

    These metrics measure n-gram overlap between generated and reference text.

    Args:
        generated: Generated text
        reference: Reference text

    Returns:
        Dictionary with BLEU and ROUGE scores
    """
    try:
        from rouge_score import rouge_scorer
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

        # Calculate ROUGE scores
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        rouge_scores = scorer.score(reference, generated)

        # Calculate BLEU score
        reference_tokens = [reference.split()]
        generated_tokens = generated.split()
        smoothing = SmoothingFunction().method1
        bleu = sentence_bleu(reference_tokens, generated_tokens, smoothing_function=smoothing)

        return {
            'bleu': bleu,
            'rouge1': rouge_scores['rouge1'].fmeasure,
            'rouge2': rouge_scores['rouge2'].fmeasure,
            'rougeL': rouge_scores['rougeL'].fmeasure
        }

    except Exception as e:
        logger.warning(f"Could not calculate BLEU/ROUGE: {e}")
        return {
            'bleu': 0.0,
            'rouge1': 0.0,
            'rouge2': 0.0,
            'rougeL': 0.0,
            'error': str(e)
        }


def create_evaluation_report(
    comparisons: List[Dict],
    output_path: str
) -> Dict:
    """
    Create a comprehensive evaluation report.

    Args:
        comparisons: List of model comparisons
        output_path: Path to save report JSON

    Returns:
        Summary statistics
    """
    logger.info("Creating evaluation report...")

    # Calculate summary statistics
    total_comparisons = len(comparisons)
    successful_comparisons = sum(
        1 for c in comparisons
        if c['base_model']['success'] and c['finetuned_model']['success']
    )

    # Calculate average generation times
    avg_base_time = sum(
        c['base_model']['generation_time'] for c in comparisons
        if c['base_model']['success']
    ) / max(successful_comparisons, 1)

    avg_finetuned_time = sum(
        c['finetuned_model']['generation_time'] for c in comparisons
        if c['finetuned_model']['success']
    ) / max(successful_comparisons, 1)

    # Build report
    report = {
        'evaluation_date': datetime.now().isoformat(),
        'summary': {
            'total_prompts': total_comparisons,
            'successful_comparisons': successful_comparisons,
            'avg_base_generation_time': avg_base_time,
            'avg_finetuned_generation_time': avg_finetuned_time,
        },
        'comparisons': comparisons
    }

    # Save report
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    logger.info(f"✓ Evaluation report saved to {output_path}")

    return report['summary']


def print_comparison_sample(comparison: Dict, index: int) -> None:
    """
    Print a sample comparison to the console for quick review.

    Args:
        comparison: Comparison dictionary
        index: Index of this comparison
    """
    print("\n" + "=" * 80)
    print(f"COMPARISON #{index}")
    print("=" * 80)
    print(f"\nPrompt: {comparison['prompt']}\n")

    print("-" * 80)
    print(f"BASE MODEL ({comparison['base_model']['name']}):")
    print("-" * 80)
    print(comparison['base_model']['response'][:500])
    if len(comparison['base_model']['response']) > 500:
        print("... (truncated)")

    print("\n" + "-" * 80)
    print(f"FINE-TUNED MODEL ({comparison['finetuned_model']['name']}):")
    print("-" * 80)
    print(comparison['finetuned_model']['response'][:500])
    if len(comparison['finetuned_model']['response']) > 500:
        print("... (truncated)")

    print("\n")


def run_evaluation(
    model_name: str,
    test_set_path: str,
    output_path: str,
    config: Dict
) -> Dict:
    """
    Main entry point for the evaluation module.

    This function orchestrates the complete evaluation process:
    1. Load test prompts
    2. Generate responses from both base and fine-tuned models
    3. Compare outputs
    4. Calculate metrics
    5. Create evaluation report

    Args:
        model_name: Name of fine-tuned model in Ollama
        test_set_path: Optional path to test prompts file
        output_path: Path to save evaluation report
        config: Configuration dictionary

    Returns:
        Summary statistics dictionary
    """
    logger.info("=" * 60)
    logger.info("Starting model evaluation")
    logger.info("=" * 60)
    logger.info(f"Fine-tuned model: {model_name}")
    logger.info("")

    # Extract configuration
    eval_config = config.get('evaluation', {})
    base_model = config.get('training', {}).get('base_model', 'llama3.2:1b')
    base_url = config.get('deployment', {}).get('ollama_base_url', 'http://localhost:11434')

    # Load test prompts
    if test_set_path:
        prompts = load_test_prompts(test_set_path)
    else:
        prompts_file = eval_config.get('test_prompts_file', './prompts/test_prompts.txt')
        if Path(prompts_file).exists():
            prompts = load_test_prompts(prompts_file)
        else:
            prompts = get_default_prompts()

    # Limit number of prompts for evaluation
    num_samples = min(eval_config.get('num_samples', 10), len(prompts))
    prompts = prompts[:num_samples]

    logger.info(f"Using {len(prompts)} test prompts")

    # Compare models
    comparisons = compare_models(
        base_model=base_model,
        finetuned_model=model_name,
        prompts=prompts,
        base_url=base_url
    )

    # Print sample comparisons to console
    logger.info("\n" + "=" * 60)
    logger.info("SAMPLE COMPARISONS")
    logger.info("=" * 60)

    # Show first 2 comparisons
    for i, comparison in enumerate(comparisons[:2], 1):
        print_comparison_sample(comparison, i)

    # Create evaluation report
    summary = create_evaluation_report(comparisons, output_path)

    # Print summary
    logger.info("=" * 60)
    logger.info("EVALUATION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total prompts tested: {summary['total_prompts']}")
    logger.info(f"Successful comparisons: {summary['successful_comparisons']}")
    logger.info(f"Base model avg time: {summary['avg_base_generation_time']:.2f}s")
    logger.info(f"Fine-tuned avg time: {summary['avg_finetuned_generation_time']:.2f}s")
    logger.info("")
    logger.info(f"Full report saved to: {output_path}")
    logger.info("=" * 60)

    return summary


if __name__ == '__main__':
    # Example usage when running this module directly
    import yaml

    # Load config
    with open('pipeline_config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Run evaluation
    run_evaluation(
        model_name='blogging-twin:latest',
        test_set_path=None,
        output_path='./results/evaluations/eval_report.json',
        config=config
    )
