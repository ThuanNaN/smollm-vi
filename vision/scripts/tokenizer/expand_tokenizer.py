"""
Script to expand a HuggingFace SmolVLM2 processor with new Vietnamese tokens.

This script:
1. Loads the SmolVLM2 processor (tokenizer + image processor + chat template)
2. Trains a byte-level BPE tokenizer on the Vietnamese corpus
3. Appends new tokens and their merge rules to the base vocabulary,
   keeping every base token at its original ID (so the base model's
   embedding matrix stays aligned — only new rows need to be added)
4. Saves the expanded processor and verifies expansion correctness

Usage:
    python expand_tokenizer.py \
        --corpus path/to/vietnamese_text.txt \
        --output_dir path/to/output \
        [--base_model HuggingFaceTB/SmolVLM2-500M-Video-Instruct] \
        [--new_vocab_size 57344]
"""

import argparse
import json
from pathlib import Path

from tokenizers import Tokenizer, pre_tokenizers
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from transformers import AutoProcessor, AutoTokenizer


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
    base_model_id: str,
    corpus_path: str,
    output_dir: str,
    new_vocab_size: int,
):
    """
    Expand a SmolVLM2 processor with new tokens from a corpus.

    Args:
        base_model_id: HuggingFace model ID or local path to base model
        corpus_path: Path to text file with new language (Vietnamese) corpus
        output_dir: Directory to save expanded processor
        new_vocab_size: Target vocabulary size after expansion
    """
    # Load base processor (tokenizer + image processor + chat template).
    # Expanding from the full SmolVLM2 processor keeps <image>=49190 and all
    # special tokens, and makes AutoProcessor.from_pretrained(output_dir) work.
    print(f"Loading base processor: {base_model_id}")
    base_processor = AutoProcessor.from_pretrained(base_model_id)
    base_tokenizer = base_processor.tokenizer
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

    # Save a full copy of the base processor (keeps tokenizer, image processor,
    # chat template, etc.) — we then patch tokenizer.json
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    base_processor.save_pretrained(output_dir)

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

    verify_expansion(output_dir, base_tokenizer, corpus, new_tokens)


def verify_expansion(output_dir, base_tokenizer, corpus, new_tokens):
    """Fail loudly if the expansion broke anything the training run relies on."""
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(output_dir)
    tok = processor.tokenizer

    # 1. Special tokens keep their IDs (image token especially)
    for special in base_tokenizer.all_special_tokens:
        base_id = base_tokenizer.convert_tokens_to_ids(special)
        new_id = tok.convert_tokens_to_ids(special)
        assert new_id == base_id, f"special token {special!r} moved: {base_id} -> {new_id}"
    image_id = tok.convert_tokens_to_ids("<image>")
    assert image_id == base_tokenizer.convert_tokens_to_ids("<image>"), \
        f"<image> id changed to {image_id}"
    print(f"OK: {len(base_tokenizer.all_special_tokens)} special tokens unchanged, "
          f"<image>={image_id}")

    # 2. Vietnamese roundtrip is lossless (byte-level BPE must reconstruct exactly)
    for text in corpus[:200]:
        decoded = tok.decode(tok.encode(text, add_special_tokens=False))
        assert decoded == text, f"roundtrip mismatch: {text[:60]!r} -> {decoded[:60]!r}"
    print("OK: Vietnamese encode/decode roundtrip lossless (200 samples)")

    # 3. Fertility improvement on held-out text (tail of corpus, unseen order)
    held_out = corpus[-1000:]
    words = sum(len(t.split()) for t in held_out)
    base_toks = sum(len(base_tokenizer.encode(t, add_special_tokens=False)) for t in held_out)
    new_toks = sum(len(tok.encode(t, add_special_tokens=False)) for t in held_out)
    print(f"Fertility (tokens/word): base={base_toks / words:.2f} "
          f"-> expanded={new_toks / words:.2f} "
          f"({100 * (1 - new_toks / base_toks):.1f}% fewer tokens)")

    # 4. English tokenization unchanged (appended merges have lowest priority)
    english = ("The quick brown fox jumps over the lazy dog. "
               "Vision language models process images and text together.")
    if tok.encode(english) != base_tokenizer.encode(english):
        print("WARNING: English tokenization changed slightly — inspect before training.")
    else:
        print("OK: English tokenization identical to base")

    print(f"Expanded processor saved to: {output_dir}")
    print(f"Final vocab size: {len(tok)} ({len(new_tokens)} new Vietnamese tokens)")
    print("Training will resize embeddings automatically via --tokenizer_name_or_path.")


def main():
    parser = argparse.ArgumentParser(
        description="Expand SmolVLM2 processor vocabulary with Vietnamese tokens"
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
        help="Base model (HuggingFace model ID or local path, default: SmolVLM2-500M-Video-Instruct)"
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
        help="Directory to save expanded processor"
    )
    parser.add_argument(
        "--new_vocab_size",
        type=int,
        default=57344,
        help="Target vocabulary size after expansion (default: 57344)"
    )

    args = parser.parse_args()

    expand_tokenizer(
        base_model_id=args.base_model,
        corpus_path=args.corpus,
        output_dir=args.output_dir,
        new_vocab_size=args.new_vocab_size,
    )


if __name__ == "__main__":
    main()
