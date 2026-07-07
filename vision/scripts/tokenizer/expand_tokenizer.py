"""
Script to expand a HuggingFace tokenizer with new tokens from a Vietnamese corpus.

This script:
1. Loads a base tokenizer (e.g., SmolLM2-360M-Instruct) and saves a copy
2. Trains a byte-level BPE tokenizer on the Vietnamese corpus
3. Appends new tokens and their merge rules to the base vocabulary,
   keeping every base token at its original ID (so the base model's
   embedding matrix stays aligned — only new rows need to be added)
4. Saves the expanded tokenizer and verifies it loads

Usage:
    python expand_tokenizer.py \
        --base_tokenizer HuggingFaceTB/SmolLM2-360M-Instruct \
        --corpus path/to/vietnamese_text.txt \
        --output_dir path/to/output \
        --new_vocab_size 57000
"""

import argparse
import json
from pathlib import Path

from tokenizers import Tokenizer, pre_tokenizers
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from transformers import AutoTokenizer


def load_corpus(corpus_path: str) -> list[str]:
    """Load text corpus from file.

    Args:
        corpus_path: Path to text file with one line per sample

    Returns:
        List of non-empty text lines
    """
    with open(corpus_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def train_vietnamese_tokenizer(corpus: list[str], vocab_size: int) -> Tokenizer:
    """Train a byte-level BPE tokenizer on the Vietnamese corpus.

    Byte-level pre-tokenization matches the representation used by
    GPT2-style tokenizers (SmolLM2 included), so the learned tokens are
    directly comparable with the base vocabulary.
    """
    tokenizer = Tokenizer(BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )
    tokenizer.train_from_iterator(corpus, trainer=trainer)
    return tokenizer


def _normalize_merges(merges: list) -> list[tuple[str, str]]:
    """Normalize merges to (left, right) tuples.

    tokenizer.json stores merges either as "left right" strings (older
    tokenizers versions) or as ["left", "right"] pairs (newer versions).
    Byte-level tokens never contain a literal space, so splitting on the
    single space is safe.
    """
    normalized = []
    for merge in merges:
        if isinstance(merge, str):
            left, right = merge.split(" ")
        else:
            left, right = merge
        normalized.append((left, right))
    return normalized


def expand_tokenizer(
    base_tokenizer_id: str,
    corpus_path: str,
    output_dir: str,
    new_vocab_size: int,
):
    """
    Expand a HuggingFace tokenizer with new tokens from a corpus.

    Args:
        base_tokenizer_id: HuggingFace model ID or local path to base tokenizer
        corpus_path: Path to text file with new language (Vietnamese) corpus
        output_dir: Directory to save expanded tokenizer
        new_vocab_size: Target vocabulary size after expansion
    """
    # Load base tokenizer
    print(f"Loading base tokenizer: {base_tokenizer_id}")
    base_tokenizer = AutoTokenizer.from_pretrained(base_tokenizer_id)
    base_vocab_size = len(base_tokenizer)

    print(f"Base tokenizer vocab size: {base_vocab_size}")
    print(f"Target vocab size: {new_vocab_size}")
    print(f"New tokens to add: {new_vocab_size - base_vocab_size}")

    if new_vocab_size <= base_vocab_size:
        raise ValueError(
            f"new_vocab_size ({new_vocab_size}) must be larger than the "
            f"base vocab size ({base_vocab_size})"
        )

    # Load corpus
    print(f"Loading corpus from: {corpus_path}")
    corpus = load_corpus(corpus_path)
    print(f"Loaded {len(corpus)} lines from corpus")

    if len(corpus) == 0:
        raise ValueError("Corpus is empty. Please check the corpus path.")

    # Save a full copy of the base tokenizer (keeps chat template, special
    # tokens map, tokenizer_config.json, etc.) — we then patch tokenizer.json
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    base_tokenizer.save_pretrained(output_dir)

    # Train a Vietnamese tokenizer to learn candidate tokens and merges.
    # Train up to the full target size so there are enough candidates even
    # after dropping ones that already exist in the base vocabulary.
    print("Training Vietnamese BPE tokenizer on corpus...")
    vi_tokenizer = train_vietnamese_tokenizer(corpus, vocab_size=new_vocab_size)
    vi_model = json.loads(vi_tokenizer.to_str())["model"]
    vi_merges = _normalize_merges(vi_model["merges"])
    print(f"Learned {len(vi_merges)} candidate merges from corpus")

    # Merge into the base tokenizer.json
    tokenizer_json_path = output_path / "tokenizer.json"
    with open(tokenizer_json_path, "r", encoding="utf-8") as f:
        tokenizer_data = json.load(f)

    vocab: dict[str, int] = tokenizer_data["model"]["vocab"]
    merges = tokenizer_data["model"]["merges"]
    merges_are_pairs = bool(merges) and not isinstance(merges[0], str)

    added_token_ids = [t["id"] for t in tokenizer_data.get("added_tokens", [])]
    next_id = max([*vocab.values(), *added_token_ids, -1]) + 1
    known_tokens = set(vocab) | {t["content"] for t in tokenizer_data.get("added_tokens", [])}

    # Append new merges in learned order: each merge's parts are guaranteed
    # to exist already (byte-level alphabet or the result of an earlier merge)
    new_tokens = []
    for left, right in vi_merges:
        if len(known_tokens) >= new_vocab_size:
            break
        token = left + right
        if token in known_tokens:
            continue
        vocab[token] = next_id
        merges.append([left, right] if merges_are_pairs else f"{left} {right}")
        known_tokens.add(token)
        new_tokens.append(token)
        next_id += 1

    with open(tokenizer_json_path, "w", encoding="utf-8") as f:
        json.dump(tokenizer_data, f, ensure_ascii=False)

    # Save the list of newly added tokens for inspection
    with open(output_path / "new_vietnamese_tokens.json", "w", encoding="utf-8") as f:
        json.dump(new_tokens, f, ensure_ascii=False, indent=2)

    # Verify the expanded tokenizer loads and base IDs are unchanged
    expanded_tokenizer = AutoTokenizer.from_pretrained(output_dir)
    sample = corpus[0][:200]
    base_ids = base_tokenizer.encode(sample)
    expanded_ids = expanded_tokenizer.encode(sample)

    print(f"Expanded tokenizer saved to: {output_dir}")
    print(f"Final vocab size: {len(expanded_tokenizer)}")
    print(f"New Vietnamese tokens added: {len(new_tokens)}")
    print(f"Sample encoding: {len(base_ids)} tokens (base) -> "
          f"{len(expanded_ids)} tokens (expanded)")
    print("Note: resize the model embeddings with "
          "model.resize_token_embeddings(len(tokenizer)) before training.")


def main():
    parser = argparse.ArgumentParser(
        description="Expand tokenizer vocabulary with Vietnamese tokens"
    )
    parser.add_argument(
        "--base_tokenizer",
        type=str,
        required=True,
        help="Base tokenizer (HuggingFace model ID or local path)"
    )
    parser.add_argument(
        "--corpus",
        type=str,
        required=True,
        help="Path to Vietnamese text corpus (one line per sample)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save expanded tokenizer"
    )
    parser.add_argument(
        "--new_vocab_size",
        type=int,
        default=57000,
        help="Target vocabulary size after expansion (default: 57000)"
    )

    args = parser.parse_args()

    expand_tokenizer(
        base_tokenizer_id=args.base_tokenizer,
        corpus_path=args.corpus,
        output_dir=args.output_dir,
        new_vocab_size=args.new_vocab_size,
    )


if __name__ == "__main__":
    main()
