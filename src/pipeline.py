"""
Pipeline Orchestration Module

This module runs the complete fine-tuning pipeline from start to finish.
It coordinates all the individual modules in the correct sequence.

Pipeline stages:
1. Data ingestion - Read blog files
2. Dataset preparation - Format for training
3. Fine-tuning - Train the model
4. Deployment - Deploy to Ollama
5. Evaluation - Test the model

Key functions:
- run_full_pipeline(): Execute all stages in sequence
- handle_pipeline_error(): Graceful error handling with recovery
"""

import logging
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime

# Import all the module functions
from .data_ingestion import run_ingestion
from .dataset_prep import run_dataset_preparation
from .finetune import run_finetuning
from .deploy import run_deployment
from .evaluate import run_evaluation

logger = logging.getLogger(__name__)


def create_pipeline_log_file() -> str:
    """
    Create a timestamped log file for this pipeline run.

    Returns:
        Path to log file
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = Path(f"./results/pipeline_run_{timestamp}.log")
    log_file.parent.mkdir(parents=True, exist_ok=True)

    return str(log_file)


def log_pipeline_stage(stage_name: str, status: str = "start"):
    """
    Log pipeline stage transitions with visual markers.

    Args:
        stage_name: Name of the pipeline stage
        status: Either 'start', 'complete', or 'failed'
    """
    if status == "start":
        logger.info("\n" + "=" * 80)
        logger.info(f"PIPELINE STAGE: {stage_name}")
        logger.info("=" * 80 + "\n")
    elif status == "complete":
        logger.info("\n" + "-" * 80)
        logger.info(f"✓ {stage_name} completed successfully")
        logger.info("-" * 80 + "\n")
    elif status == "failed":
        logger.error("\n" + "-" * 80)
        logger.error(f"✗ {stage_name} failed")
        logger.error("-" * 80 + "\n")


def run_full_pipeline(
    blog_dir: str,
    model_name: str,
    config: Dict,
    skip_evaluation: bool = False
) -> Dict:
    """
    Run the complete fine-tuning pipeline.

    This function orchestrates all stages of the pipeline:
    1. Ingest blog files from directory
    2. Prepare training dataset
    3. Fine-tune the model
    4. Deploy to Ollama
    5. (Optional) Evaluate performance

    Args:
        blog_dir: Directory containing blog files
        model_name: Name for the deployed model
        config: Configuration dictionary
        skip_evaluation: Whether to skip evaluation step

    Returns:
        Dictionary with pipeline results and paths
    """
    logger.info("\n")
    logger.info("╔" + "=" * 78 + "╗")
    logger.info("║" + " " * 20 + "LLM BLOGGING TWIN - FULL PIPELINE" + " " * 25 + "║")
    logger.info("╚" + "=" * 78 + "╝")
    logger.info(f"\nPipeline started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Blog directory: {blog_dir}")
    logger.info(f"Target model name: {model_name}")
    logger.info(f"Skip evaluation: {skip_evaluation}\n")

    # Store results from each stage
    results = {
        'pipeline_start_time': datetime.now().isoformat(),
        'config': config,
        'stages': {}
    }

    try:
        # ================================================================
        # STAGE 1: DATA INGESTION
        # ================================================================
        log_pipeline_stage("1. Data Ingestion", "start")

        ingestion_output = Path(config['data']['output_dir']) / 'raw_blogs.jsonl'
        num_blogs = run_ingestion(
            input_dir=blog_dir,
            output_path=str(ingestion_output),
            config=config
        )

        results['stages']['ingestion'] = {
            'status': 'complete',
            'num_blogs': num_blogs,
            'output_path': str(ingestion_output)
        }

        log_pipeline_stage("1. Data Ingestion", "complete")

        if num_blogs == 0:
            raise ValueError("No blogs were ingested! Check your input directory.")

        # ================================================================
        # STAGE 2: DATASET PREPARATION
        # ================================================================
        log_pipeline_stage("2. Dataset Preparation", "start")

        training_data_output = Path(config['data']['output_dir']) / 'training_data.jsonl'
        train_size, val_size = run_dataset_preparation(
            input_path=str(ingestion_output),
            output_path=str(training_data_output),
            split_ratio=config['data']['train_split'],
            config=config
        )

        results['stages']['dataset_prep'] = {
            'status': 'complete',
            'train_size': train_size,
            'val_size': val_size,
            'output_path': str(training_data_output)
        }

        log_pipeline_stage("2. Dataset Preparation", "complete")

        if train_size == 0:
            raise ValueError("No training examples were created!")

        # ================================================================
        # STAGE 3: FINE-TUNING
        # ================================================================
        log_pipeline_stage("3. Model Fine-tuning", "start")

        model_output = Path(config['training']['output_dir']) / 'blogging_twin'
        model_path = run_finetuning(
            training_data_path=str(training_data_output),
            output_dir=str(model_output),
            config=config
        )

        results['stages']['finetuning'] = {
            'status': 'complete',
            'model_path': model_path
        }

        log_pipeline_stage("3. Model Fine-tuning", "complete")

        # ================================================================
        # STAGE 4: DEPLOYMENT
        # ================================================================
        log_pipeline_stage("4. Model Deployment", "start")

        deployed_name = run_deployment(
            model_path=model_path,
            model_name=model_name,
            config=config
        )

        results['stages']['deployment'] = {
            'status': 'complete',
            'model_name': deployed_name
        }

        log_pipeline_stage("4. Model Deployment", "complete")

        # ================================================================
        # STAGE 5: EVALUATION (Optional)
        # ================================================================
        if not skip_evaluation:
            log_pipeline_stage("5. Model Evaluation", "start")

            eval_output = Path(config['evaluation']['output_dir']) / 'eval_report.json'
            eval_summary = run_evaluation(
                model_name=deployed_name,
                test_set_path=None,
                output_path=str(eval_output),
                config=config
            )

            results['stages']['evaluation'] = {
                'status': 'complete',
                'summary': eval_summary,
                'report_path': str(eval_output)
            }

            log_pipeline_stage("5. Model Evaluation", "complete")
        else:
            logger.info("\nSkipping evaluation as requested.\n")
            results['stages']['evaluation'] = {
                'status': 'skipped'
            }

        # ================================================================
        # PIPELINE COMPLETE
        # ================================================================
        results['pipeline_end_time'] = datetime.now().isoformat()
        results['status'] = 'success'

        logger.info("\n")
        logger.info("╔" + "=" * 78 + "╗")
        logger.info("║" + " " * 25 + "PIPELINE COMPLETED SUCCESSFULLY!" + " " * 22 + "║")
        logger.info("╚" + "=" * 78 + "╝")
        logger.info(f"\nPipeline finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"\n✓ Your blogging twin is ready!")
        logger.info(f"  Model name: {model_name}")
        logger.info(f"  Test it with: ollama run {model_name}")
        logger.info(f"\n  Or start the web interface:")
        logger.info(f"    python cli.py serve\n")

        return results

    except Exception as e:
        # Log the error
        logger.error(f"\n\n{'=' * 80}")
        logger.error("PIPELINE FAILED")
        logger.error(f"{'=' * 80}")
        logger.error(f"Error: {str(e)}")
        logger.error(f"{'=' * 80}\n")

        # Store error information
        results['pipeline_end_time'] = datetime.now().isoformat()
        results['status'] = 'failed'
        results['error'] = str(e)

        # Re-raise the exception
        raise


def run_partial_pipeline(
    start_stage: str,
    blog_dir: str,
    model_name: str,
    config: Dict
) -> Dict:
    """
    Run pipeline starting from a specific stage.

    This is useful for recovery if a stage fails or for iterative development.

    Args:
        start_stage: Stage to start from ('prep', 'finetune', 'deploy', 'evaluate')
        blog_dir: Directory containing blog files
        model_name: Name for the deployed model
        config: Configuration dictionary

    Returns:
        Dictionary with pipeline results
    """
    logger.info(f"Starting partial pipeline from stage: {start_stage}")

    # Map stage names to their starting points
    # This allows resuming the pipeline from any stage
    if start_stage == 'prep':
        # Start from dataset preparation
        # Assumes ingestion was already completed
        ingestion_output = Path(config['data']['output_dir']) / 'raw_blogs.jsonl'
        if not ingestion_output.exists():
            raise FileNotFoundError(f"Ingestion output not found: {ingestion_output}")

        # Continue with dataset prep...
        # (Implementation similar to full pipeline)
        pass

    elif start_stage == 'finetune':
        # Start from fine-tuning
        # Assumes data prep was already completed
        pass

    elif start_stage == 'deploy':
        # Start from deployment
        # Assumes fine-tuning was already completed
        pass

    elif start_stage == 'evaluate':
        # Just run evaluation
        # Assumes model is already deployed
        pass

    else:
        raise ValueError(f"Unknown stage: {start_stage}")

    # TODO: Implement partial pipeline logic
    logger.warning("Partial pipeline is not fully implemented yet")
    logger.warning("Please use the full pipeline for now")


if __name__ == '__main__':
    # Example usage when running this module directly
    import yaml

    # Load config
    with open('pipeline_config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Run full pipeline
    results = run_full_pipeline(
        blog_dir='./data/raw',
        model_name='blogging-twin:latest',
        config=config,
        skip_evaluation=False
    )

    print(f"\nPipeline results: {results}")
