"""
Build a Vietnamese tokenizer-training corpus from locally cached HF datasets.

Corpus = Wikipedia articles (base vocabulary) + train-split text from
ViOCRVQA / OpenViVQA / UIT-ViIC repeated --vqa_repeat times (the VQA text is
~10MB vs ~2.5GB of wiki, so without repetition BPE would never learn its
domain vocabulary). Only train splits are used — eval text must not
influence the tokenizer.

Usage:
    python build_tokenizer_corpus.py \
        --output data/vietnamese_corpus.txt \
        --wiki_articles 300000 --vqa_repeat 20
"""

import argparse
import json
import zipfile
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import hf_hub_download


def iter_wiki_texts(max_articles: int | None):
    ds = load_dataset("vietgpt/wikipedia_vi", split="train")
    n = len(ds) if max_articles is None else min(max_articles, len(ds))
    for i in range(n):
        text = (ds[i].get("text") or "").strip()
        if text:
            yield text.replace("\n", " ")


def viocrvqa_train_lines() -> list[str]:
    zip_path = hf_hub_download(
        "huyhuy123/ViOCRVQA", "data_ViOCRVQA.zip", repo_type="dataset"
    )
    with zipfile.ZipFile(zip_path) as zf:
        data = json.load(zf.open("data/train.json"))
    lines = []
    for ann in data["annotations"]:
        q = (ann.get("question") or "").strip()
        if q:
            lines.append(q)
        for a in ann.get("answers") or []:
            a = (a or "").strip()
            if a:
                lines.append(a)
    return lines


def openvivqa_train_lines() -> list[str]:
    json_path = hf_hub_download(
        "uitnlp/OpenViVQA-dataset", "vlsp2023_train_data.json", repo_type="dataset"
    )
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    lines = []
    for ann in data["annotations"].values():
        for key in ("question", "answer"):
            v = (ann.get(key) or "").strip()
            if v:
                lines.append(v)
    return lines


def uitviic_train_lines() -> list[str]:
    ds = load_dataset("ThucPD/UIT-ViIC", split="train")
    lines = []
    for row in ds:
        try:
            convs = json.loads(row["conversations"])
        except (TypeError, json.JSONDecodeError):
            continue
        for turn in convs:
            v = (turn.get("value") or "").replace("<image>", "").strip()
            if v:
                lines.append(v)
    return lines


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--wiki_articles", type=int, default=300_000)
    parser.add_argument("--vqa_repeat", type=int, default=20)
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Cap wiki articles AND vqa lines (smoke tests)")
    args = parser.parse_args()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    wiki_cap = args.max_samples or args.wiki_articles
    vqa_lines = viocrvqa_train_lines() + openvivqa_train_lines() + uitviic_train_lines()
    if args.max_samples:
        vqa_lines = vqa_lines[: args.max_samples]
    print(f"VQA/caption lines: {len(vqa_lines)} (repeated x{args.vqa_repeat})")

    wiki_count = 0
    with open(out, "w", encoding="utf-8") as f:
        for text in iter_wiki_texts(wiki_cap):
            f.write(text + "\n")
            wiki_count += 1
        for _ in range(args.vqa_repeat):
            for line in vqa_lines:
                f.write(line + "\n")

    total = wiki_count + len(vqa_lines) * args.vqa_repeat
    print(f"Wrote {total} lines ({wiki_count} wiki + "
          f"{len(vqa_lines)}x{args.vqa_repeat} vqa) to {out}")


if __name__ == "__main__":
    main()
