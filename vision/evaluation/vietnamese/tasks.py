"""
Vietnamese evaluation tasks for SmolVLM benchmarking.

Custom tasks for:
- Vietnamese VQA (accuracy metric)
- Vietnamese Captioning (BLEU, ROUGE, CIDEr metrics)
- Vietnamese OCR-VQA (accuracy metric)
"""

from dataclasses import dataclass, field
from typing import Any, List


@dataclass
class VietnameseVQATask:
    """Vietnamese Visual Question Answering task."""
    name: str = "vietnamese_vqa"
    dataset_path: str = "path/to/vietnamese-vqa-test"
    metric: str = "accuracy"

    def process_prediction(self, prediction: str, ground_truth: str) -> bool:
        """Check if prediction matches ground truth.

        Args:
            prediction: Model's predicted answer
            ground_truth: Ground truth answer

        Returns:
            True if prediction matches ground truth
        """
        # Handle Vietnamese text normalization
        return prediction.strip().lower() == ground_truth.strip().lower()


@dataclass
class VietnameseCaptionTask:
    """Vietnamese Image Captioning task."""
    name: str = "vietnamese_caption"
    dataset_path: str = "path/to/vietnamese-caption-test"
    metrics: List[str] = field(default_factory=lambda: ["bleu", "rouge", "cider"])


@dataclass
class VietnameseOCRVQATask:
    """Vietnamese OCR Visual Question Answering task."""
    name: str = "vietnamese_ocr_vqa"
    dataset_path: str = "path/to/vietnamese-ocr-vqa-test"
    metric: str = "accuracy"

    def process_prediction(self, prediction: str, ground_truth: str) -> bool:
        """Check if prediction matches ground truth for OCR-VQA."""
        # OCR-VQA may need more flexible matching
        pred = prediction.strip().lower()
        gt = ground_truth.strip().lower()
        return pred == gt


# Task registry
VIETNAMESE_TASKS = {
    "vietnamese_vqa": VietnameseVQATask,
    "vietnamese_caption": VietnameseCaptionTask,
    "vietnamese_ocr_vqa": VietnameseOCRVQATask,
}


def get_task(task_name: str) -> Any:
    """Get a task by name.

    Args:
        task_name: Name of the task

    Returns:
        Task instance
    """
    if task_name not in VIETNAMESE_TASKS:
        raise ValueError(f"Unknown task: {task_name}. Available: {list(VIETNAMESE_TASKS.keys())}")
    return VIETNAMESE_TASKS[task_name]()
