# vision/evaluation/vietnamese/run_evaluation.py

import argparse
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

# Assume these are available via relative imports after the tasks directory setup
from .tasks import (
    EvaluationTask,
    TextBaselineTask,
    ImageFusionTask, # Placeholder for image-based task
    VIETNAMESE_PROMPTS
)

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class ResourceCache:
    """
    Manages and caches expensive resources like models and processors
    to ensure they are loaded only once per execution run.
    """
    _model = None
    _processor = None

    @classmethod
    def load_resources(cls, model_path: str) -> Tuple[Optional[Any], Optional[Any]]:
        """Loads and returns the shared model and processor instances."""
        if cls._model is not None and cls._processor is not None:
            logging.info("Resources already cached. Skipping expensive reload.")
            return cls._model, cls._processor

        logging.info(f"Attempting to load core resources from path: {model_path}")
        try:
            # --- START Model Loading Logic (Must match tasks.py) ---
            from transformers import AutoProcessor
            from transformers.models.idefics3 import Idefics3ForConditionalGeneration
            import torch

            processor = AutoProcessor.from_pretrained(
                model_path, trust_remote_code=True,
            )
            model = Idefics3ForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
            ).cuda() # Assuming CUDA is available

            model.eval()
            # --- END Model Loading Logic ---
            logging.info("Core resources loaded and cached successfully.")
            cls._model = model
            cls._processor = processor
            return cls._model, cls._processor

        except Exception as e:
            logging.error(f"CRITICAL ERROR: Failed to load core resources from {model_path}. Reason: {e}")
            # Clear cached state on failure
            cls._model = None
            cls._processor = None
            return None, None


def discover_tasks() -> List[type['EvaluationTask']]:
    """
    Dynamically discovers all concrete EvaluationTask subclasses available
    in the current module.
    """
    # In a real-world scenario, we would use 'inspect' to read definitions.
    # For simplicity and reliability in this scaffold, we list known tasks.
    discovered_tasks = [
        TextBaselineTask,
        ImageFusionTask,
    ]
    return discovered_tasks


def main(args: Optional[Any]) -> None:
    """Main entry point for running the entire evaluation suite."""
    logging.info("--- Starting Vietnamese Evaluation Suite Runner ---")

    # 1. Load shared resources first
    model, processor = ResourceCache.load_resources(args.model_path)
    if model is None or processor is None:
        logging.error("Cannot proceed with evaluation due to failed resource loading.")
        return

    # 2. Discover all tasks
    TaskClass = discover_tasks()
    logging.info(f"Discovered {len(TaskClass)} evaluation tasks.")

    all_results: List[Dict[str, Any]] = []

    # 3. Iterate and Execute (The resilient loop)
    for TaskClass in TaskClass:
        task_name = TaskClass.__name__
        logging.info(f"\n{'='*60}")
        logging.info(f"ATTEMPTING TO RUN TASK: {task_name} ({TaskClass.category})")
        logging.info(f"{'='*60}\n")

        try:
            # Instantiate the task
            task_instance = TaskClass()

            # Execute the run method, passing shared resources and any unique args
            # This assumes all tasks accept (model, processor) plus optional model-specific args.
            results = task_instance.run(model=model, processor=processor, max_new_tokens=args.max_new_tokens or 128)

            all_results.append({
                "task": task_name,
                "status": "SUCCESS",
                "data": results
            })
            logging.info(f"Successfully completed and collected results for {task_name}.")

        except Exception as e:
            # CRITICAL: Catching the exception ensures the suite continues even if one task fails dramatically.
            error_message = f"Execution failed for {task_name} with unhandled error: {type(e).__name__}: {str(e)}"
            logging.error(error_message)
            all_results.append({
                "task": task_name,
                "status": "FAILED",
                "error": str(e)
            })

    # 4. Final Report Generation
    print("\n\n===============================================")
    print("       EVALUATION SUITE SUMMARY REPORT        ")
    print("===============================================")
    success_count = sum(1 for res in all_results if res['status'] == 'SUCCESS')
    fail_count = len(all_results) - success_count
    print(f"Total Tasks Run: {len(all_results)}")
    print(f"Successful Runs: {success_count}")
    print(f"Failed Runs: {fail_count}\n")

    # Add detailed output analysis here... (e.g., summarizing metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified runner for SmolVLM Vietnamese Evaluation Suite.")
    parser.add_argument("--model_path",
                        type=str,
                        required=True,
                        help="Path to the HuggingFace model directory (e.g., HuggingFaceTB/SmolVLM-500M-Instruct)")
    parser.add_argument("--max_new_tokens",
                        type=int,
                        default=128,
                        help="Maximum number of tokens to generate per prompt.")
    args = parser.parse_args()

    main(args)