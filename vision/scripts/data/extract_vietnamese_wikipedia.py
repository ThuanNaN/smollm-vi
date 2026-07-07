"""
Extract Vietnamese text from Wikipedia for tokenizer training.

Usage:
    python extract_vietnamese_wikipedia.py \
        --input vietgpt/wikipedia_vi \
        --output data/vietnamese_corpus.txt

The --input argument accepts a HuggingFace dataset ID, a local dataset
directory, or a HF hub cache path (e.g. .../datasets--vietgpt--wikipedia_vi,
which is resolved back to its dataset ID so the cached copy is reused).
"""

import argparse
from pathlib import Path


def resolve_dataset_path(input_path: str) -> str:
    """Resolve a HF hub cache path back to its dataset ID.

    A path like /data/.cache/huggingface/hub/datasets--vietgpt--wikipedia_vi
    cannot be passed to load_dataset() directly, but the equivalent dataset ID
    (vietgpt/wikipedia_vi) can — and load_dataset will reuse the same cache.
    Other inputs (dataset IDs, plain local directories) pass through unchanged.
    """
    name = Path(input_path).name
    if name.startswith("datasets--"):
        return name.removeprefix("datasets--").replace("--", "/")
    return input_path


def main():
    parser = argparse.ArgumentParser(
        description="Extract Vietnamese text from Wikipedia"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="vietgpt/wikipedia_vi",
        help="Dataset ID, local dataset directory, or HF hub cache path"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/vietnamese_corpus.txt",
        help="Output file path"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to process (None = all)"
    )

    args = parser.parse_args()

    # Import here to avoid issues when script is imported
    from datasets import load_dataset

    dataset_path = resolve_dataset_path(args.input)
    print(f"Loading Wikipedia dataset from: {dataset_path}")
    ds = load_dataset(dataset_path)
    print(f"Loaded dataset: {ds}")

    # Prefer the train split, otherwise use the first available one
    split_name = "train" if "train" in ds else list(ds.keys())[0]
    print(f"Using split: {split_name}")

    total_samples = len(ds[split_name])
    if args.max_samples:
        total_samples = min(args.max_samples, total_samples)
        print(f"Processing {total_samples} samples (max_samples={args.max_samples})")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Writing to: {args.output}")
    count = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for i, sample in enumerate(ds[split_name]):
            if args.max_samples and count >= args.max_samples:
                break

            # Extract text content
            text = sample.get("text", "")
            if text and text.strip():
                f.write(text.strip() + "\n")
                count += 1

            if (i + 1) % 100000 == 0:
                print(f"  Processed {i + 1} samples, wrote {count} texts...")

    print(f"Done! Wrote {count} samples to {args.output}")


if __name__ == "__main__":
    main()
