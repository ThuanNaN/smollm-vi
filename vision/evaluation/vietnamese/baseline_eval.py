"""
Baseline evaluation script for Vietnamese SmolVLM benchmarking.

Evaluates SmolVLM-500M-Instruct on Vietnamese VQA and captioning tasks
BEFORE fine-tuning.

Usage:
    python baseline_eval.py --model_path HuggingFaceTB/SmolVLM-500M-Instruct
"""

import argparse
import io
import json
from datetime import datetime
from pathlib import Path

import torch
from datasets import load_dataset
from PIL import Image
from transformers import AutoProcessor, Idefics3ForConditionalGeneration


def load_image_from_path(image_path: str) -> Image.Image:
    """Load image from a local path or URL. Returns None on failure."""
    try:
        if image_path.startswith(("http://", "https://")):
            import requests
            response = requests.get(image_path, timeout=10)
            response.raise_for_status()
            return Image.open(io.BytesIO(response.content)).convert("RGB")
        return Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Warning: Could not load image {image_path}: {e}")
        return None


def resolve_image(sample) -> Image.Image:
    """Get a PIL image from a dataset sample's image field."""
    image = sample.get("image")
    if image is None:
        return None
    if isinstance(image, str):
        return load_image_from_path(image)
    if image.mode != "RGB":
        image = image.convert("RGB")
    return image


def extract_qa(sample) -> tuple[str, str]:
    """Extract (question, answer) from a sample.

    Supports plain question/answer columns (e.g. ViOCRVQA) as well as
    llava-style `conversations` (string-encoded JSON or list of dicts
    with from=human/gpt), as used by UIT-ViIC.
    """
    question = sample.get("question")
    answer = sample.get("answer") or sample.get("answers")
    if isinstance(answer, list):
        answer = answer[0] if answer else None
    if question and answer:
        return str(question).strip(), str(answer).strip()

    conversations = sample.get("conversations", "")
    try:
        conv_list = json.loads(conversations) if isinstance(conversations, str) else conversations
        question, answer = "", ""
        for conv in conv_list or []:
            if conv.get("from") == "human":
                question = conv.get("value", "").replace("<image>", "").strip()
            elif conv.get("from") == "gpt":
                answer = conv.get("value", "").strip()
        return question, answer
    except Exception:
        return "", ""


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


def generate_answer(model, processor, image, question, max_new_tokens=256):
    """Run one image+text generation and return the decoded answer."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": question}
            ]
        }
    ]

    prompt = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False
    )

    inputs = processor(
        text=prompt,
        images=[image],
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False
        )

    output_ids = generated_ids[0][len(inputs.input_ids[0]):]
    return processor.decode(output_ids, skip_special_tokens=True).strip()


def evaluate_vqa(model, processor, dataset, num_samples=None):
    """Evaluate model on a Vietnamese VQA dataset.

    Args:
        model: Loaded VLM model
        processor: Model processor
        dataset: HuggingFace dataset with image + question/answer (or
            llava-style conversations) columns
        num_samples: Number of samples to evaluate (None = all)

    Returns:
        Dictionary with evaluation metrics
    """
    results = []
    correct = 0
    total = 0

    if num_samples:
        dataset = dataset.select(range(min(num_samples, len(dataset))))

    print(f"Evaluating on {len(dataset)} samples...")

    for i, sample in enumerate(dataset):
        image = resolve_image(sample)
        if image is None:
            continue

        question, answer = extract_qa(sample)
        if not question or not answer:
            continue

        predicted_answer = generate_answer(model, processor, image, question)

        # Check if correct (exact match, case-insensitive)
        is_correct = predicted_answer.lower() == answer.strip().lower()
        if is_correct:
            correct += 1

        total += 1
        results.append({
            "image_id": i,
            "question": question,
            "predicted": predicted_answer,
            "ground_truth": answer,
            "correct": is_correct
        })

        if (i + 1) % 10 == 0:
            current_acc = correct / total * 100 if total else 0
            print(f"  Sample {i + 1}/{len(dataset)} | Acc: {current_acc:.1f}%")

    accuracy = correct / total * 100 if total > 0 else 0

    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "samples": results
    }


def evaluate_captioning(model, processor, dataset, num_samples=None):
    """Evaluate model on a Vietnamese image captioning dataset.

    Args:
        model: Loaded VLM model
        processor: Model processor
        dataset: HuggingFace dataset with image + caption (or llava-style
            conversations) columns
        num_samples: Number of samples to evaluate

    Returns:
        Dictionary with BLEU, ROUGE metrics
    """
    from rouge_score import rouge_scorer

    results = []
    predictions = []
    references = []

    if num_samples:
        dataset = dataset.select(range(min(num_samples, len(dataset))))

    print(f"Evaluating captioning on {len(dataset)} samples...")

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False)

    for i, sample in enumerate(dataset):
        image = resolve_image(sample)
        if image is None:
            continue

        caption = sample.get("caption")
        if not caption:
            _, caption = extract_qa(sample)
        if not caption:
            continue
        caption = str(caption).strip()

        predicted = generate_answer(
            model, processor, image,
            "Mô tả hình ảnh này bằng tiếng Việt.",
            max_new_tokens=128,
        )

        rouge_scores = scorer.score(caption, predicted)

        predictions.append(predicted)
        references.append(caption)

        results.append({
            "image_id": i,
            "predicted": predicted,
            "ground_truth": caption,
            "rouge1": rouge_scores["rouge1"].fmeasure,
            "rougeL": rouge_scores["rougeL"].fmeasure
        })

        if (i + 1) % 5 == 0:
            print(f"  Sample {i + 1}/{len(dataset)}")

    # Calculate BLEU (optional dependency)
    try:
        import sacrebleu
        bleu = sacrebleu.corpus_bleu(predictions, [references]).score if predictions else 0.0
    except ImportError:
        print("Warning: sacrebleu not installed, skipping BLEU (pip install sacrebleu)")
        bleu = None

    avg_rouge1 = sum(r["rouge1"] for r in results) / len(results) if results else 0
    avg_rougeL = sum(r["rougeL"] for r in results) / len(results) if results else 0

    return {
        "bleu": bleu,
        "rouge1": avg_rouge1 * 100,
        "rougeL": avg_rougeL * 100,
        "total": len(results),
        "samples": results
    }


def load_dataset_with_fallback(dataset_id: str, preferred_split: str = "test"):
    """Load a dataset, falling back across common split names."""
    for split in (preferred_split, "validation", "train"):
        try:
            ds = load_dataset(dataset_id, split=split, trust_remote_code=True)
            print(f"Loaded {dataset_id} [{split}]: {len(ds)} samples")
            print(f"Columns: {ds.column_names}")
            return ds
        except Exception as e:
            last_error = e
    print(f"Could not load {dataset_id}: {last_error}")
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Baseline evaluation for Vietnamese SmolVLM"
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
        "--num_samples",
        type=int,
        default=50,
        help="Number of samples to evaluate per task"
    )
    parser.add_argument(
        "--skip_captioning",
        action="store_true",
        help="Skip captioning evaluation (UIT-ViIC)"
    )
    parser.add_argument(
        "--skip_vqa",
        action="store_true",
        help="Skip OCR-VQA evaluation (ViOCRVQA)"
    )

    args = parser.parse_args()

    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load model
    model, processor = load_model_and_processor(args.model_path)

    # Results dictionary
    all_results = {
        "model": args.model_path,
        "date": datetime.now().isoformat(),
        "tasks": {}
    }

    print("\n" + "=" * 60)
    print("Vietnamese Evaluation")
    print("=" * 60)

    # UIT-ViIC captioning
    if not args.skip_captioning:
        print("\nLoading UIT-ViIC dataset...")
        uit_viic = load_dataset_with_fallback("ThucPD/UIT-ViIC")
        if uit_viic is not None:
            print("\n" + "=" * 60)
            print("UIT-ViIC Evaluation (Captioning)")
            print("=" * 60)
            try:
                caption_results = evaluate_captioning(
                    model, processor, uit_viic,
                    num_samples=args.num_samples
                )
                all_results["tasks"]["uit_viic"] = {
                    k: caption_results[k] for k in ("bleu", "rouge1", "rougeL", "total")
                }
                all_results["tasks"]["uit_viic"]["samples"] = caption_results["samples"]
                bleu = caption_results["bleu"]
                bleu_str = f"{bleu:.1f}" if bleu is not None else "n/a"
                print(f"\nUIT-ViIC - BLEU: {bleu_str}, ROUGE-L: {caption_results['rougeL']:.1f}")
            except Exception as e:
                print(f"UIT-ViIC evaluation error: {e}")
                all_results["tasks"]["uit_viic"] = {"error": str(e)}

    # ViOCRVQA visual question answering
    if not args.skip_vqa:
        print("\nLoading ViOCRVQA dataset...")
        viocrvqa = load_dataset_with_fallback("huyhuy123/ViOCRVQA")
        if viocrvqa is not None:
            print("\n" + "=" * 60)
            print("ViOCRVQA Evaluation (OCR-VQA)")
            print("=" * 60)
            try:
                vqa_results = evaluate_vqa(
                    model, processor, viocrvqa,
                    num_samples=args.num_samples
                )
                all_results["tasks"]["viocrvqa"] = {
                    k: vqa_results[k] for k in ("accuracy", "correct", "total")
                }
                all_results["tasks"]["viocrvqa"]["samples"] = vqa_results["samples"]
                print(f"\nViOCRVQA - Accuracy: {vqa_results['accuracy']:.1f}%")
            except Exception as e:
                print(f"ViOCRVQA evaluation error: {e}")
                all_results["tasks"]["viocrvqa"] = {"error": str(e)}

    # Save results
    results_file = output_path / "baseline_results.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print("Evaluation Complete")
    print("=" * 60)
    print(f"Results saved to: {results_file}")

    # Print summary
    print("\nSummary:")
    for task_name, task_results in all_results["tasks"].items():
        if "error" in task_results:
            print(f"  {task_name}: ERROR - {task_results['error']}")
            continue
        if "accuracy" in task_results:
            print(f"  {task_name}: Accuracy = {task_results['accuracy']:.1f}%")
        if "bleu" in task_results:
            bleu = task_results["bleu"]
            bleu_str = f"{bleu:.1f}" if bleu is not None else "n/a"
            print(f"  {task_name}: BLEU = {bleu_str}, ROUGE-L = {task_results['rougeL']:.1f}")


if __name__ == "__main__":
    main()
