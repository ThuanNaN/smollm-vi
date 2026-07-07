"""
Simple Vietnamese text baseline evaluation for SmolVLM-500M-Instruct.

This script tests the model's Vietnamese language capabilities WITHOUT images,
to establish a baseline before fine-tuning.

Usage:
    python text_baseline.py --model_path HuggingFaceTB/SmolVLM-500M-Instruct
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import torch
from transformers import AutoProcessor, Idefics3ForConditionalGeneration


# Vietnamese test prompts with expected answer types
VIETNAMESE_PROMPTS = [
    {
        "question": "Xin chào, bạn tên gì?",
        "category": "greeting"
    },
    {
        "question": "Việt Nam có những thành phố lớn nào?",
        "category": "geography"
    },
    {
        "question": "Hãy giải thích quy trình quang hợp.",
        "category": "science"
    },
    {
        "question": "Công thức tính diện tích hình tròn là gì?",
        "category": "math"
    },
    {
        "question": "Ai là tác giả của Truyện Kiều?",
        "category": "literature"
    },
    {
        "question": "Năm 2024 là năm gì?",
        "category": "general"
    },
    {
        "question": "Màu cờ tổ quốc Việt Nam là màu gì?",
        "category": "general"
    },
    {
        "question": "Hãy viết một bài thơ ngắn về mùa xuân.",
        "category": "creative"
    },
]


def load_model_and_processor(model_path: str):
    """Load model and processor for inference."""
    print(f"Loading model: {model_path}")

    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Idefics3ForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(device)

    model.eval()
    print(f"Model loaded successfully on {device}")

    return model, processor


def evaluate_text_generation(model, processor, prompts, max_new_tokens=128):
    """Evaluate model on Vietnamese text prompts.

    Args:
        model: Loaded model
        processor: Model processor
        prompts: List of prompt dictionaries
        max_new_tokens: Maximum tokens to generate

    Returns:
        List of results
    """
    results = []

    print(f"\nEvaluating on {len(prompts)} Vietnamese prompts...")
    print("=" * 70)

    for i, prompt_data in enumerate(prompts):
        question = prompt_data["question"]
        category = prompt_data["category"]

        print(f"\n{i+1}. [{category}] {question}")
        print("-" * 50)

        # SmolVLM's chat template expects content as a list of typed chunks
        messages = [
            {"role": "user", "content": [{"type": "text", "text": question}]}
        ]
        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = processor(text=text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

        output_ids = generated_ids[0][len(inputs.input_ids[0]):]
        response = processor.decode(output_ids, skip_special_tokens=True)

        print(f"   Response: {response[:150]}...")

        results.append({
            "question": question,
            "category": category,
            "response": response,
            "response_length": len(response)
        })

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Vietnamese text baseline evaluation for SmolVLM"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="HuggingFaceTB/SmolVLM-500M-Instruct",
        help="Path to the model"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="evals/vietnamese/baseline",
        help="Output directory for results"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=128,
        help="Maximum tokens to generate"
    )

    args = parser.parse_args()

    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load model
    model, processor = load_model_and_processor(args.model_path)

    # Evaluate
    results = evaluate_text_generation(
        model, processor, VIETNAMESE_PROMPTS,
        max_new_tokens=args.max_new_tokens
    )

    # Save results
    results_file = output_path / "text_baseline_results.json"
    output_data = {
        "model": args.model_path,
        "date": datetime.now().isoformat(),
        "results": results
    }

    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 70)
    print("Evaluation Complete")
    print("=" * 70)
    print(f"Results saved to: {results_file}")


if __name__ == "__main__":
    main()
