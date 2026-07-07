"""
Convert ViOCRVQA dataset to webdataset format for m4 training.

Each shard is a flat tar containing {key}.jpg, {key}.txt and {key}.json
members (the standard webdataset layout, matching
convert_vietnamese_to_webdataset.py).

Usage:
    python convert_viocrvqa.py --output_dir data/webdataset/vietnamese_ocr --max_samples 1000
"""

import argparse
import io
import json
import tarfile
from pathlib import Path

from datasets import load_dataset


def build_text(sample: dict) -> str:
    """Format the question/answer pair as chat-style text."""
    lines = []
    question = sample.get("question")
    if question:
        lines.append(f"[user]: {str(question).strip()}")
    answer = sample.get("answer") or sample.get("answers")
    if isinstance(answer, list):
        answer = answer[0] if answer else None
    if answer:
        lines.append(f"[assistant]: {str(answer).strip()}")
    return "\n".join(lines)


def sample_to_webdataset(sample: dict, sample_id: int) -> dict:
    """Convert a single sample to webdataset member files.

    Args:
        sample: Dataset sample dictionary
        sample_id: Unique identifier used as the webdataset key

    Returns:
        Dictionary mapping member filenames to bytes
    """
    key = f"{sample_id:06d}"
    result = {}

    image = sample["image"]
    if image.mode != "RGB":
        image = image.convert("RGB")
    img_buffer = io.BytesIO()
    image.save(img_buffer, format="JPEG", quality=85)
    result[f"{key}.jpg"] = img_buffer.getvalue()

    text = build_text(sample)
    if text:
        result[f"{key}.txt"] = text.encode("utf-8")

    metadata = {
        "id": str(sample.get("id", sample_id)),
        "source": "ViOCRVQA",
        "question": sample.get("question"),
        "answer": sample.get("answer") or sample.get("answers"),
    }
    result[f"{key}.json"] = json.dumps(metadata, ensure_ascii=False).encode("utf-8")

    return result


def write_shard(output_path: Path, shard_num: int, samples: list):
    """Write a shard file.

    Args:
        output_path: Output directory
        shard_num: Shard number
        samples: List of dicts mapping member filenames to bytes
    """
    shard_path = output_path / f"{shard_num:05d}.tar"

    with tarfile.open(shard_path, "w") as tar:
        for sample_data in samples:
            for filename, data in sample_data.items():
                tarinfo = tarfile.TarInfo(name=filename)
                tarinfo.size = len(data)
                tar.addfile(tarinfo, io.BytesIO(data))


def convert_viocrvqa_to_webdataset(
    output_dir: str,
    shard_size: int = 500,
    max_samples: int = None,
):
    """Convert ViOCRVQA to webdataset format.

    Args:
        output_dir: Output directory for shards
        shard_size: Samples per shard
        max_samples: Maximum samples (None = all)
    """
    print("Loading ViOCRVQA dataset...")
    ds = load_dataset("huyhuy123/ViOCRVQA", split="train", trust_remote_code=True)

    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))

    print(f"Loaded {len(ds)} samples")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    shard_num = 0
    shard_samples = []
    successful = 0
    failed = 0

    for i in range(len(ds)):
        try:
            # Indexing decodes the image; corrupted samples raise here
            sample = ds[i]
            if sample.get("image") is None:
                failed += 1
                continue

            shard_samples.append(sample_to_webdataset(sample, i))
            successful += 1

        except Exception as e:
            print(f"  Error processing sample {i}: {e}")
            failed += 1
            continue

        # Write shard when full
        if len(shard_samples) >= shard_size:
            write_shard(output_path, shard_num, shard_samples)
            shard_num += 1
            shard_samples = []
            print(f"  Written shard {shard_num}")

        # Progress
        if (i + 1) % 5000 == 0:
            print(f"  Processed {i + 1}/{len(ds)} samples...")

    # Write final shard
    if shard_samples:
        write_shard(output_path, shard_num, shard_samples)
        shard_num += 1

    print(f"\nConversion complete!")
    print(f"  Total shards: {shard_num}")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"  Output: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert ViOCRVQA to webdataset format"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for webdataset shards"
    )
    parser.add_argument(
        "--shard_size",
        type=int,
        default=500,
        help="Samples per shard (default: 500)"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum samples to process"
    )

    args = parser.parse_args()
    convert_viocrvqa_to_webdataset(
        output_dir=args.output_dir,
        shard_size=args.shard_size,
        max_samples=args.max_samples,
    )


if __name__ == "__main__":
    main()
