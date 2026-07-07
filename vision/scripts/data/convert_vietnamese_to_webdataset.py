"""
Convert Vietnamese datasets to webdataset format for m4 training.

Handles UIT-ViIC and ViOCRVQA datasets with image downloading.

Usage:
    python convert_vietnamese_to_webdataset.py \\
        --dataset ThucPD/UIT-ViIC \\
        --output_dir data/webdataset/vietnamese_caption \\
        --shard_size 500
"""

import argparse
import io
import json
import os
import tarfile
from pathlib import Path

import requests
from PIL import Image
from datasets import load_dataset


def download_image(image_path: str) -> Image.Image:
    """Download and load image from path/URL.

    Args:
        image_path: Local path or URL to image

    Returns:
        PIL Image or None if failed
    """
    try:
        if image_path.startswith(('http://', 'https://')):
            # Download from URL
            response = requests.get(image_path, timeout=10)
            return Image.open(io.BytesIO(response.content)).convert('RGB')
        else:
            # Local path
            if os.path.exists(image_path):
                return Image.open(image_path).convert('RGB')
            else:
                print(f"  Warning: Image not found: {image_path}")
                return None
    except Exception as e:
        print(f"  Warning: Could not load image {image_path}: {e}")
        return None


def sample_to_webdataset(sample: dict, sample_id: int) -> dict:
    """Convert a single sample to webdataset format.

    Args:
        sample: Dataset sample dictionary
        sample_id: Unique identifier

    Returns:
        Dictionary with .jpg, .txt, .json keys and bytes values
    """
    result = {}

    # Handle image
    image_data = sample.get('image')
    if image_data:
        # Download or load image
        if isinstance(image_data, str):
            image = download_image(image_data)
        else:
            image = image_data

        if image:
            img_buffer = io.BytesIO()
            image.save(img_buffer, format='JPEG', quality=85)
            img_buffer.seek(0)
            result[f'{sample_id:06d}.jpg'] = img_buffer.getvalue()

    # Handle text (conversations or caption/answer)
    conversations = sample.get('conversations', '')
    if conversations:
        # Parse and format conversations
        if isinstance(conversations, str):
            try:
                conv_list = json.loads(conversations)
                # Format as chat
                text_lines = []
                for conv in conv_list:
                    role = 'user' if conv.get('from') == 'human' else 'assistant'
                    value = conv.get('value', '').replace('<image>', '').strip()
                    if value:
                        text_lines.append(f'[{role}]: {value}')
                text = '\n'.join(text_lines)
            except:
                text = conversations
        else:
            text = str(conversations)
        result[f'{sample_id:06d}.txt'] = text.encode('utf-8')
    else:
        # Try caption or answer fields
        caption = sample.get('caption', '')
        if caption:
            result[f'{sample_id:06d}.txt'] = str(caption).encode('utf-8')
        answer = sample.get('answer', '')
        if answer:
            result[f'{sample_id:06d}.txt'] = str(answer).encode('utf-8')

    # Metadata
    metadata = {
        'id': sample.get('id', str(sample_id)),
        'source': sample.get('source', 'unknown'),
    }
    result[f'{sample_id:06d}.json'] = json.dumps(metadata).encode('utf-8')

    return result


def convert_dataset(
    dataset_id: str,
    output_dir: str,
    shard_size: int = 500,
    split: str = 'train',
    max_samples: int = None,
):
    """Convert a dataset to webdataset format.

    Args:
        dataset_id: HuggingFace dataset ID
        output_dir: Output directory for shards
        shard_size: Samples per shard
        split: Dataset split
        max_samples: Maximum samples to process (None = all)
    """
    print(f"\n{'='*60}")
    print(f"Converting: {dataset_id} (split={split})")
    print(f"{'='*60}")

    # Load dataset
    ds = load_dataset(dataset_id, split=split, trust_remote_code=True)

    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))

    print(f"Loaded {len(ds)} samples")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    shard_num = 0
    samples_in_shard = []
    total_images = 0
    failed_images = 0

    for i, sample in enumerate(ds):
        sample_data = sample_to_webdataset(sample, i)
        samples_in_shard.append((i, sample_data))

        if 'jpg' in str(sample_data):
            total_images += 1
        if not any(k.endswith('.jpg') for k in sample_data.keys()):
            failed_images += 1

        # Write shard when full
        if len(samples_in_shard) >= shard_size:
            write_shard(output_path, shard_num, samples_in_shard)
            shard_num += 1
            samples_in_shard = []
            print(f"  Written shard {shard_num} ({shard_size} samples)")

        # Progress update
        if (i + 1) % 1000 == 0:
            print(f"  Processed {i + 1}/{len(ds)} samples...")

    # Write final shard
    if samples_in_shard:
        write_shard(output_path, shard_num, samples_in_shard)
        shard_num += 1

    print(f"\nConversion complete!")
    print(f"  Total shards: {shard_num}")
    print(f"  Total samples: {sum(len(s[1]) for s in samples_in_shard) if samples_in_shard else 0}")
    print(f"  Images processed: {total_images}")
    print(f"  Output dir: {output_dir}")


def write_shard(output_path: Path, shard_num: int, samples: list):
    """Write a shard file.

    Args:
        output_path: Output directory
        shard_num: Shard number
        samples: List of (id, data_dict) tuples
    """
    shard_path = output_path / f'{shard_num:05d}.tar'

    with tarfile.open(shard_path, 'w') as tar:
        for sample_id, sample_data in samples:
            for filename, data in sample_data.items():
                tarinfo = tarfile.TarInfo(name=filename)
                tarinfo.size = len(data)
                tar.addfile(tarinfo, io.BytesIO(data))


def main():
    parser = argparse.ArgumentParser(
        description='Convert Vietnamese dataset to webdataset format'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        help='HuggingFace dataset ID (e.g., ThucPD/UIT-ViIC)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Output directory for webdataset shards'
    )
    parser.add_argument(
        '--shard_size',
        type=int,
        default=500,
        help='Samples per shard (default: 500)'
    )
    parser.add_argument(
        '--split',
        type=str,
        default='train',
        help='Dataset split (default: train)'
    )
    parser.add_argument(
        '--max_samples',
        type=int,
        default=None,
        help='Maximum samples to process (default: all)'
    )

    args = parser.parse_args()
    convert_dataset(
        dataset_id=args.dataset,
        output_dir=args.output_dir,
        shard_size=args.shard_size,
        split=args.split,
        max_samples=args.max_samples,
    )


if __name__ == '__main__':
    main()