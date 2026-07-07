# Vietnamese SmolVLM2 Adaptation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Adapt SmolVLM2-500M-Video-Instruct to Vietnamese: expand the tokenizer (+8k Vietnamese tokens), convert cached Vietnamese+English datasets to smolvlm2's llava-JSON format, train stage-1 (image) with LoRA + trainable embeddings on 1×24GB, and benchmark before/after on Vietnamese VQA/captioning.

**Architecture:** Everything builds on the `vision/smolvlm2` stack (HF Trainer). Data prep scripts read directly from the local HF hub cache and emit llava-JSON + images under `$DATA_FOLDER`. The tokenizer is expanded by appending BPE merges to the base processor's tokenizer.json (base IDs unchanged). The trainer gets three minimal changes (bug fix, expanded-tokenizer loading with mean-init embedding resize, `modules_to_save`). A standalone eval harness in `vision/evaluation/vietnamese/` runs the same tasks against the base model and the finetuned checkpoint.

**Tech Stack:** transformers (AutoProcessor / AutoModelForImageTextToText), tokenizers (BPE), peft (LoRA + modules_to_save), datasets, pyarrow, rouge_score, sacrebleu, pytest (for pure-logic tests only — repo has no test suite).

## Global Constraints

- Base model: `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` (vocab 49280, hidden 960, untied embeddings, `<image>` id 49190).
- Tokenizer target size: **57344** (49280 + 8064, multiple of 128). Base token IDs must never change.
- Hardware: 1× GPU 24GB → per-device batch 1–2, grad accum to effective batch 8, bf16, gradient checkpointing.
- All dataset reads must hit the existing HF cache (`~/.cache/huggingface`) — everything needed is already cached; never re-download.
- Train splits only in the tokenizer corpus and training data. Eval splits (ViOCRVQA test, OpenViVQA dev, UIT-ViIC test/valid) are reserved for the harness.
- Mixture YAML top-level must be a **dict** (e.g. `vietnamese_stage1: [...]`) — `build_datasets` iterates `.items()`; a top-level list crashes.
- llava-JSON sample format: `{"image": "images/x.jpg", "conversations": [{"from": "human", "value": "<image>\n..."}, {"from": "gpt", "value": "..."}]}`; `<image>` appears exactly once, in the first human turn; text-only samples have no `image` key.
- Python env: the one used for `vision/smolvlm2` (`cd vision/smolvlm2 && pip install -e .`); scripts under `vision/scripts` and `vision/evaluation` additionally need `rouge_score`, `sacrebleu`, `pyarrow`, `pytest`.
- Commit after every task; messages end with `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.

---

### Task 1: Delete the m4-track files

The repo currently mixes two training stacks. Per the approved spec, the m4-track files are removed; smolvlm2 is the only stack.

**Files:**
- Delete: `vision/experiments/pretraining/vietnamese/config_stage1_1gpu.yaml`
- Delete: `vision/experiments/pretraining/vietnamese/accelerate_config_1gpu.yaml`
- Delete: `vision/experiments/pretraining/vietnamese/ds_config_zero2.json`
- Delete: `vision/scripts/data/convert_vietnamese_to_webdataset.py`
- Delete: `vision/scripts/data/convert_viocrvqa.py`
- Delete: `vision/m4/pyproject.toml`
- Modify: `vision/m4/__init__.py` (revert to original)

**Interfaces:**
- Consumes: nothing.
- Produces: a clean tree where only smolvlm2-track files remain; later tasks create their replacements.

- [ ] **Step 1: Delete files and revert `vision/m4/__init__.py`**

```bash
cd /home/thuannd/Repository/smollm-vi
git rm vision/experiments/pretraining/vietnamese/config_stage1_1gpu.yaml \
       vision/experiments/pretraining/vietnamese/accelerate_config_1gpu.yaml \
       vision/experiments/pretraining/vietnamese/ds_config_zero2.json \
       vision/scripts/data/convert_vietnamese_to_webdataset.py \
       vision/scripts/data/convert_viocrvqa.py \
       vision/m4/pyproject.toml
git checkout HEAD~1 -- vision/m4/__init__.py 2>/dev/null || true
```

Then verify `vision/m4/__init__.py` contains exactly:

```python
from m4.utils import logging
```

If the `git checkout` did not restore it (the file was modified in the index, not a past commit), overwrite it with that single line manually.

- [ ] **Step 2: Verify nothing else references the deleted files**

Run: `grep -rn "config_stage1_1gpu\|ds_config_zero2\|convert_vietnamese_to_webdataset\|convert_viocrvqa" --include="*.py" --include="*.sh" --include="*.yaml" --include="*.md" . | grep -v docs/superpowers`
Expected: only `vision/experiments/pretraining/vietnamese/train_1gpu.sh` comments and `CLAUDE.md` may match — both get rewritten in later tasks; no other code references.

- [ ] **Step 3: Commit**

```bash
git add -A vision/m4 vision/experiments vision/scripts
git commit -m "Remove m4-track Vietnamese files; consolidate on smolvlm2 stack

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 2: Tokenizer corpus builder (`build_tokenizer_corpus.py`)

Replaces `extract_vietnamese_wikipedia.py`. Corpus = wiki articles + train-split VQA/caption text repeated `--vqa_repeat` times (VQA text is only ~10MB vs wiki ~2.5GB, so without upweighting BPE would ignore it).

**Files:**
- Delete: `vision/scripts/data/extract_vietnamese_wikipedia.py`
- Create: `vision/scripts/tokenizer/build_tokenizer_corpus.py`

**Interfaces:**
- Consumes: HF cache datasets `vietgpt/wikipedia_vi`, `huyhuy123/ViOCRVQA` (zip), `uitnlp/OpenViVQA-dataset` (json), `ThucPD/UIT-ViIC` (parquet).
- Produces: a plain-text corpus file (one sample per line), consumed by Task 3 via `--corpus`. CLI: `python build_tokenizer_corpus.py --output <path> [--wiki_articles 300000] [--vqa_repeat 20] [--max_samples N]`.

- [ ] **Step 1: Write the script**

```python
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
```

- [ ] **Step 2: Smoke-run against the real cache**

Run:
```bash
cd /home/thuannd/Repository/smollm-vi
python vision/scripts/tokenizer/build_tokenizer_corpus.py \
    --output /tmp/claude-1000/-home-thuannd-Repository-smollm-vi/*/scratchpad/corpus_smoke.txt \
    --max_samples 500 --vqa_repeat 2
```
Expected: prints `VQA/caption lines: 500 (repeated x2)` and `Wrote 1500 lines (500 wiki + 500x2 vqa)`. Inspect a few lines: `head -3 <output>` shows Vietnamese text, no `<image>` markers.

- [ ] **Step 3: Delete the old script and commit**

```bash
git rm vision/scripts/data/extract_vietnamese_wikipedia.py
git add vision/scripts/tokenizer/build_tokenizer_corpus.py
git commit -m "Add tokenizer corpus builder (wiki + upweighted VQA text)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 3: Rework `expand_tokenizer.py` (base = SmolVLM2 processor, built-in verification)

Keep the existing merge-append core (it is correct). Change: expand from the SmolVLM2 **processor** and save the full processor dir (so `AutoProcessor.from_pretrained(<dir>)` works in training/eval); default target 57344; add verification.

**Files:**
- Modify: `vision/scripts/tokenizer/expand_tokenizer.py`

**Interfaces:**
- Consumes: corpus file from Task 2.
- Produces: expanded processor directory (tokenizer.json + processor/image-processor configs + chat template + `new_vietnamese_tokens.json`). Consumed by Task 5 (`--tokenizer_name_or_path`) and Task 8 (eval `--processor_path`). CLI: `python expand_tokenizer.py --corpus <txt> --output_dir <dir> [--base_model HuggingFaceTB/SmolVLM2-500M-Video-Instruct] [--new_vocab_size 57344]`.

- [ ] **Step 1: Apply the following changes to `expand_tokenizer.py`**

Keep `load_corpus`, `train_vietnamese_tokenizer`, `_normalize_merges`, and the merge-append loop in `expand_tokenizer()` unchanged. Make these edits:

1. Module docstring: replace the usage block with the new CLI (`--base_model`, default vocab 57344).
2. Imports: add `from transformers import AutoProcessor` (keep `AutoTokenizer`).
3. In `expand_tokenizer()` replace the base-loading and save-copy section:

```python
    # Load base processor (tokenizer + image processor + chat template).
    # Expanding from the full SmolVLM2 processor keeps <image>=49190 and all
    # special tokens, and makes AutoProcessor.from_pretrained(output_dir) work.
    print(f"Loading base processor: {base_model_id}")
    base_processor = AutoProcessor.from_pretrained(base_model_id)
    base_tokenizer = base_processor.tokenizer
    base_vocab_size = len(base_tokenizer)
```

and save with `base_processor.save_pretrained(output_dir)` instead of `base_tokenizer.save_pretrained(output_dir)` (the tokenizer.json patching that follows is unchanged).

4. Rename the parameter `base_tokenizer_id` → `base_model_id` (and the CLI flag `--base_tokenizer` → `--base_model`, default `"HuggingFaceTB/SmolVLM2-500M-Video-Instruct"`); change `--new_vocab_size` default to `57344`.
5. Replace the final verification block (everything after writing `new_vietnamese_tokens.json`) with:

```python
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
```

- [ ] **Step 2: Smoke-run with the small corpus from Task 2**

Run (small target so it finishes in seconds — smoke only):
```bash
python vision/scripts/tokenizer/expand_tokenizer.py \
    --corpus <scratchpad>/corpus_smoke.txt \
    --output_dir <scratchpad>/tokenizer_smoke \
    --new_vocab_size 49800
```
Expected output includes: `OK: ... special tokens unchanged, <image>=49190`, `OK: Vietnamese encode/decode roundtrip lossless`, a fertility line showing a reduction, and `Final vocab size: 49800`. Also check the dir: `ls <scratchpad>/tokenizer_smoke` contains `tokenizer.json`, `tokenizer_config.json`, `preprocessor_config.json`, `processor_config.json`, `chat_template.json` (or the template inside tokenizer_config), `new_vietnamese_tokens.json`.

- [ ] **Step 3: Commit**

```bash
git add vision/scripts/tokenizer/expand_tokenizer.py
git commit -m "Expand tokenizer from SmolVLM2 processor with built-in verification

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 4: Dataset converter (`convert_to_llava_json.py`)

One script, one `--source` per run, all reading from HF cache and writing llava-JSON + images under `--output_dir` (= the future `$DATA_FOLDER`).

**Files:**
- Create: `vision/scripts/data/convert_to_llava_json.py`

**Interfaces:**
- Consumes: cached datasets (see Global Constraints).
- Produces under `$DATA_FOLDER`: `viocrvqa/images/*.jpg` + `viocrvqa_train.json`, `openvivqa/images/*.jpg` + `openvivqa_train.json`, `uitviic/images/*.jpg` + `uitviic_train.json`, `cauldron/<subset>/images/*.jpg` + `cauldron_<subset>.json`, `viwiki_text.json`. These JSON paths are wired into the mixture in Task 6.
- CLI: `python convert_to_llava_json.py --source {viocrvqa,openvivqa,uitviic,cauldron,viwiki} --output_dir $DATA_FOLDER [--subset vqav2] [--max_samples N]`.

- [ ] **Step 1: Write the script**

```python
"""
Convert cached Vietnamese/English datasets to smolvlm2 llava-JSON format.

Output layout under --output_dir (the training $DATA_FOLDER):
    viocrvqa/images/*.jpg   viocrvqa_train.json
    openvivqa/images/*.jpg  openvivqa_train.json
    uitviic/images/*.jpg    uitviic_train.json
    cauldron/<subset>/images/*.jpg  cauldron_<subset>.json
    viwiki_text.json        (text-only samples, no images)

Sample format (multi-turn; <image> only in the first human turn):
    {"image": "images/x.jpg",
     "conversations": [{"from": "human", "value": "<image>\\n<question>"},
                       {"from": "gpt", "value": "<answer>"}, ...]}

VQA sources group all QA pairs of one image into a single multi-turn
conversation (the_cauldron convention): the image is encoded once per
conversation instead of once per QA pair.
"""

import argparse
import io
import json
import zipfile
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import hf_hub_download


def qa_to_conversations(qa_pairs: list[tuple[str, str]]) -> list[dict]:
    """[(q, a), ...] -> llava turns; <image> only in the first human turn."""
    conversations = []
    for i, (q, a) in enumerate(qa_pairs):
        prefix = "<image>\n" if i == 0 else ""
        conversations.append({"from": "human", "value": prefix + q})
        conversations.append({"from": "gpt", "value": a})
    return conversations


def group_annotations_by_image(annotations) -> dict:
    """COCO-style annotations (list or dict of {image_id, question, answer(s)})
    -> {image_id: [(q, a), ...]} keeping only pairs with non-empty q and a."""
    if isinstance(annotations, dict):
        annotations = list(annotations.values())
    grouped: dict = {}
    for ann in annotations:
        q = (ann.get("question") or "").strip()
        answer = ann.get("answer")
        if answer is None:
            answers = ann.get("answers") or []
            answer = answers[0] if answers else None
        a = (str(answer) if answer is not None else "").strip()
        if q and a:
            grouped.setdefault(ann["image_id"], []).append((q, a))
    return grouped


def zip_basename_index(zf: zipfile.ZipFile) -> dict:
    """basename -> member name, for zips with unknown inner directory layout."""
    index = {}
    for name in zf.namelist():
        if ".ipynb_checkpoints" in name or name.endswith("/"):
            continue
        index[Path(name).name] = name
    return index


def extract_image(zf, index, filename, images_dir) -> str | None:
    """Extract one image from a zip to images_dir; returns relative path or None."""
    member = index.get(filename)
    if member is None:
        return None
    target = images_dir / filename
    if not target.exists():
        with zf.open(member) as src:
            target.write_bytes(src.read())
    return f"images/{filename}"


def write_json(samples, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False)
    print(f"Wrote {len(samples)} samples -> {path}")


def convert_viocrvqa(out_root: Path, max_samples):
    zip_path = hf_hub_download("huyhuy123/ViOCRVQA", "data_ViOCRVQA.zip",
                               repo_type="dataset")
    images_dir = out_root / "viocrvqa" / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        data = json.load(zf.open("data/train.json"))
        id_to_file = {img["id"]: img["filename"] for img in data["images"]}
        grouped = group_annotations_by_image(data["annotations"])
        index = zip_basename_index(zf)
        samples = []
        for image_id, qa_pairs in grouped.items():
            if max_samples and len(samples) >= max_samples:
                break
            filename = id_to_file.get(image_id)
            if not filename:
                continue
            rel = extract_image(zf, index, filename, images_dir)
            if rel is None:
                continue
            samples.append({"image": rel,
                            "conversations": qa_to_conversations(qa_pairs)})
    write_json(samples, out_root / "viocrvqa_train.json")


def convert_openvivqa(out_root: Path, max_samples):
    json_path = hf_hub_download("uitnlp/OpenViVQA-dataset",
                                "vlsp2023_train_data.json", repo_type="dataset")
    zip_path = hf_hub_download("uitnlp/OpenViVQA-dataset", "train-images.zip",
                               repo_type="dataset")
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    # images is a dict {id_str: filename}; annotations reference int image_id
    id_to_file = {int(k): v for k, v in data["images"].items()}
    grouped = group_annotations_by_image(data["annotations"])
    images_dir = out_root / "openvivqa" / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    samples = []
    with zipfile.ZipFile(zip_path) as zf:
        index = zip_basename_index(zf)
        for image_id, qa_pairs in grouped.items():
            if max_samples and len(samples) >= max_samples:
                break
            filename = id_to_file.get(int(image_id))
            if not filename:
                continue
            rel = extract_image(zf, index, filename, images_dir)
            if rel is None:
                continue
            samples.append({"image": rel,
                            "conversations": qa_to_conversations(qa_pairs)})
    write_json(samples, out_root / "openvivqa_train.json")


def convert_uitviic(out_root: Path, max_samples):
    ds = load_dataset("ThucPD/UIT-ViIC", split="train")
    images_dir = out_root / "uitviic" / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    samples = []
    for row in ds:
        if max_samples and len(samples) >= max_samples:
            break
        try:
            convs = json.loads(row["conversations"])
        except (TypeError, json.JSONDecodeError):
            continue
        if not convs:
            continue
        image = row["image"]  # PIL image from parquet
        if image is None:
            continue
        filename = f"{row['id']}.jpg"
        target = images_dir / filename
        if not target.exists():
            image.convert("RGB").save(target, format="JPEG", quality=90)
        # UIT-ViIC puts <image> at the END of the first human turn; normalize
        # it to the leading position the smolvlm2 dataset expects.
        first = convs[0]["value"].replace("<image>", "").strip()
        convs[0]["value"] = "<image>\n" + first
        samples.append({"image": f"images/{filename}", "conversations": convs})
    write_json(samples, out_root / "uitviic_train.json")


def convert_cauldron(out_root: Path, subset: str, max_samples):
    ds = load_dataset("HuggingFaceM4/the_cauldron", subset, split="train")
    images_dir = out_root / "cauldron" / subset / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    samples = []
    for i, row in enumerate(ds):
        if max_samples and len(samples) >= max_samples:
            break
        images = row.get("images") or []
        texts = row.get("texts") or []
        if len(images) != 1 or not texts:  # keep it single-image, like our VN data
            continue
        qa_pairs = [((t.get("user") or "").strip(), (t.get("assistant") or "").strip())
                    for t in texts]
        qa_pairs = [(q, a) for q, a in qa_pairs if q and a]
        if not qa_pairs:
            continue
        filename = f"{i:07d}.jpg"
        target = images_dir / filename
        if not target.exists():
            images[0].convert("RGB").save(target, format="JPEG", quality=90)
        samples.append({"image": f"images/{filename}",
                        "conversations": qa_to_conversations(qa_pairs)})
    write_json(samples, out_root / f"cauldron_{subset}.json")


VIWIKI_PROMPT = "Hãy viết một đoạn văn bằng tiếng Việt về chủ đề: {title}"


def convert_viwiki(out_root: Path, max_samples, max_chars=1500):
    ds = load_dataset("vietgpt/wikipedia_vi", split="train")
    n = min(max_samples or 50_000, len(ds))
    samples = []
    for i in range(n):
        row = ds[i]
        text = (row.get("text") or "").strip()
        title = (row.get("title") or "").strip()
        if not text or not title or len(text) < 200:
            continue
        samples.append({"conversations": [
            {"from": "human", "value": VIWIKI_PROMPT.format(title=title)},
            {"from": "gpt", "value": text[:max_chars]},
        ]})
    write_json(samples, out_root / "viwiki_text.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True,
                        choices=["viocrvqa", "openvivqa", "uitviic",
                                 "cauldron", "viwiki"])
    parser.add_argument("--output_dir", required=True,
                        help="Training DATA_FOLDER root")
    parser.add_argument("--subset", default=None,
                        help="the_cauldron subset (required for --source cauldron)")
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()

    out_root = Path(args.output_dir)
    if args.source == "viocrvqa":
        convert_viocrvqa(out_root, args.max_samples)
    elif args.source == "openvivqa":
        convert_openvivqa(out_root, args.max_samples)
    elif args.source == "uitviic":
        convert_uitviic(out_root, args.max_samples)
    elif args.source == "cauldron":
        if not args.subset:
            parser.error("--subset is required for --source cauldron")
        convert_cauldron(out_root, args.subset, args.max_samples)
    elif args.source == "viwiki":
        convert_viwiki(out_root, args.max_samples)


if __name__ == "__main__":
    main()
```

Note for the implementer: `wikipedia_vi` may not have a `title` column — check `ds.column_names` first; if there is no `title`, derive one from the first line of `text` (`title = text.split("\n")[0][:80]`) and adjust `convert_viwiki` accordingly.

- [ ] **Step 2: Smoke-run every source with `--max_samples 20`**

Run each of these; every one must print `Wrote N samples -> ...` with N ≥ 1:
```bash
SCRATCH=<scratchpad>/data_smoke
python vision/scripts/data/convert_to_llava_json.py --source viocrvqa  --output_dir $SCRATCH --max_samples 20
python vision/scripts/data/convert_to_llava_json.py --source openvivqa --output_dir $SCRATCH --max_samples 20
python vision/scripts/data/convert_to_llava_json.py --source uitviic   --output_dir $SCRATCH --max_samples 20
python vision/scripts/data/convert_to_llava_json.py --source cauldron --subset vqav2 --output_dir $SCRATCH --max_samples 20
python vision/scripts/data/convert_to_llava_json.py --source viwiki   --output_dir $SCRATCH --max_samples 20
```

- [ ] **Step 3: Validate output structure**

Run this check:
```bash
python - <<'EOF'
import json, glob
from pathlib import Path
root = Path("<scratchpad>/data_smoke")
for jp in sorted(root.glob("*.json")):
    samples = json.load(open(jp))
    s = samples[0]
    convs = s["conversations"]
    assert convs[0]["from"] == "human" and convs[1]["from"] == "gpt", jp
    if "image" in s:
        assert (root / jp.stem.split("_")[0] if not jp.stem.startswith("cauldron")
                else root).exists()
        assert convs[0]["value"].startswith("<image>\n"), (jp, convs[0])
        assert sum(t["value"].count("<image>") for t in convs) == 1, jp
    else:
        assert "<image>" not in convs[0]["value"], jp
    print(f"{jp.name}: {len(samples)} samples OK; sample0 turns={len(convs)}")
EOF
```
Expected: 5 lines of `... OK`, no assertion errors. Manually eyeball one Vietnamese sample for mojibake (must show proper diacritics: `ạ ế ộ`...). Also open one extracted jpg to confirm it is a valid image.

- [ ] **Step 4: Commit**

```bash
git add vision/scripts/data/convert_to_llava_json.py
git commit -m "Add llava-JSON converter for Vietnamese + cauldron datasets

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 5: smolvlm2 trainer support for expanded tokenizer + PEFT fix

Three minimal changes to the smolvlm2 stack. **Do not refactor anything else in these files.**

**Files:**
- Modify: `vision/smolvlm2/smolvlm/train/args.py`
- Modify: `vision/smolvlm2/smolvlm/train/train.py`

**Interfaces:**
- Consumes: expanded processor dir from Task 3.
- Produces: `--tokenizer_name_or_path <dir>` (ModelArguments) and `--lora_modules_to_save embed_tokens lm_head` (TrainingArguments) flags used by Task 6's launch script. Function `resize_embeddings_for_expanded_tokenizer(model, model_args) -> None` in train.py.

- [ ] **Step 1: Add the two arguments in `args.py`**

In `ModelArguments` (after `padding_side`):

```python
    tokenizer_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "Load processor/tokenizer from this path instead of "
                          "model_name_or_path (e.g. an expanded-vocab processor). "
                          "Embeddings are resized + mean-initialized automatically."}
    )
```

In `TrainingArguments` (after `use_dora`):

```python
    lora_modules_to_save: List[str] = field(
        default_factory=lambda: [],
        metadata={"help": "Modules to fully train and save alongside LoRA "
                          "(e.g. embed_tokens lm_head after vocab expansion)."}
    )
```

- [ ] **Step 2: Fix the PEFT bug and pass modules_to_save in `train.py`**

Line 323: `model = apply_peft_if_needed(model, training_args)` → `model = apply_peft(model, training_args)` (the function defined at line 225; the current name crashes every `--peft_enable True` run with NameError).

In `apply_peft`, extend the `LoraConfig(...)` call:

```python
    lora_config = LoraConfig(
        r=training_args.lora_rank,
        lora_alpha=training_args.lora_alpha,
        lora_dropout=training_args.lora_dropout,
        target_modules=peft_target_modules,
        modules_to_save=training_args.lora_modules_to_save or None,
        bias=training_args.lora_bias,  # "none"/"all"/"lora_only"
        task_type="CAUSAL_LM",
    )
```

- [ ] **Step 3: Add embedding resize + mean-init in `train.py`**

New function after `prepare_model` (module level):

```python
def resize_embeddings_for_expanded_tokenizer(model, model_args):
    """If an expanded tokenizer is configured, grow the embedding matrix and
    mean-initialize each new row from the base-tokenizer encoding of the new
    token's surface text. Applies to both embed_tokens and lm_head (untied)."""
    if not model_args.tokenizer_name_or_path:
        return
    from transformers import AutoTokenizer
    new_tok = AutoTokenizer.from_pretrained(model_args.tokenizer_name_or_path)
    old_vocab = model.get_input_embeddings().weight.shape[0]
    new_len = len(new_tok)
    if new_len <= old_vocab:
        logger.info("Tokenizer size %d <= embedding rows %d; no resize needed.",
                    new_len, old_vocab)
        return
    base_tok = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    logger.info("Resizing embeddings %d -> %d and mean-initializing %d new rows",
                old_vocab, new_len, new_len - old_vocab)
    model.resize_token_embeddings(new_len)
    with torch.no_grad():
        in_emb = model.get_input_embeddings().weight
        out_emb = model.get_output_embeddings().weight
        for idx in range(old_vocab, new_len):
            text = new_tok.decode([idx])
            sub_ids = base_tok.encode(text, add_special_tokens=False)
            sub_ids = [s for s in sub_ids if s < old_vocab]
            if not sub_ids:
                continue  # keep HF's default init for undecodable byte fragments
            ids = torch.tensor(sub_ids, device=in_emb.device)
            in_emb[idx] = in_emb[ids].mean(dim=0)
            out_emb[idx] = out_emb[ids].mean(dim=0)
```

Call it in `train()` immediately after `model = prepare_model(model_args, training_args)`:

```python
    model = prepare_model(model_args, training_args)
    resize_embeddings_for_expanded_tokenizer(model, model_args)
```

(Must run before `set_trainable_params` / `apply_peft` so `modules_to_save` wraps the already-resized modules.)

- [ ] **Step 4: Point the processor at the expanded dir**

In `train()` step 5, both `AutoProcessor.from_pretrained(...)` and `SmolLMMProcessor.from_pretrained(...)` calls: replace the first positional argument `model_args.model_name_or_path` with:

```python
            model_args.tokenizer_name_or_path or model_args.model_name_or_path,
```

- [ ] **Step 5: CPU smoke-test the resize + PEFT path**

This verifies arg parsing, the NameError fix, resize and mean-init without needing the GPU (uses the smoke tokenizer from Task 3):

```bash
cd vision/smolvlm2
python - <<'EOF'
import torch
from smolvlm.train.args import ModelArguments, TrainingArguments
from smolvlm.train.train import resize_embeddings_for_expanded_tokenizer, apply_peft

class M(torch.nn.Module):
    def __init__(s, vocab=49280, hid=64):
        super().__init__()
        s.emb = torch.nn.Embedding(vocab, hid); s.head = torch.nn.Linear(hid, vocab, bias=False)
        s.q_proj = torch.nn.Linear(hid, hid)
    def get_input_embeddings(s): return s.emb
    def get_output_embeddings(s): return s.head
    def resize_token_embeddings(s, n):
        e = torch.nn.Embedding(n, s.emb.weight.shape[1]); e.weight.data[:s.emb.num_embeddings] = s.emb.weight.data
        h = torch.nn.Linear(s.emb.weight.shape[1], n, bias=False); h.weight.data[:s.head.out_features] = s.head.weight.data
        s.emb, s.head = e, h

ma = ModelArguments(model_name_or_path="HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
                    tokenizer_name_or_path="<scratchpad>/tokenizer_smoke")
m = M()
resize_embeddings_for_expanded_tokenizer(m, ma)
assert m.emb.num_embeddings == 49800, m.emb.num_embeddings
new_row = m.emb.weight[49290]
assert not torch.allclose(new_row, torch.zeros_like(new_row)), "mean-init did not run"
print("resize + mean-init OK:", m.emb.num_embeddings)
EOF
```
Expected: `resize + mean-init OK: 49800`. Then confirm the NameError fix: `grep -n "apply_peft" smolvlm/train/train.py` shows only the definition and the fixed call (no `apply_peft_if_needed`).

- [ ] **Step 6: Commit**

```bash
git add vision/smolvlm2/smolvlm/train/args.py vision/smolvlm2/smolvlm/train/train.py
git commit -m "smolvlm2: expanded-tokenizer support (resize+mean-init), fix apply_peft NameError

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 6: Mixture YAML + rewritten `train_1gpu.sh`

**Files:**
- Create: `vision/smolvlm2/scripts/mixtures/vietnamese_stage1.yaml`
- Modify: `vision/experiments/pretraining/vietnamese/train_1gpu.sh` (full rewrite)

**Interfaces:**
- Consumes: llava-JSONs from Task 4, expanded processor from Task 3, trainer flags from Task 5.
- Produces: launchable training. The template uses the literal placeholder `__DATA_FOLDER__`, resolved by the launch script into `$OUTPUT_DIR/mixture_resolved.yaml`.

- [ ] **Step 1: Write the mixture template**

`vision/smolvlm2/scripts/mixtures/vietnamese_stage1.yaml` — **top level must be a dict** (`build_datasets` iterates `.items()`):

```yaml
# Vietnamese stage-1 mixture. __DATA_FOLDER__ is replaced by train_1gpu.sh.
# Shares: ~55% Vietnamese multimodal / ~30% English (cauldron) / ~15% Vi text.
vietnamese_stage1:
  - name: viocrvqa
    json_path: __DATA_FOLDER__/viocrvqa_train.json
    path: viocrvqa
    modality: image
    sampling_strategy: all           # ~19.7k multi-turn conversations
  - name: openvivqa
    json_path: __DATA_FOLDER__/openvivqa_train.json
    path: openvivqa
    modality: image
    sampling_strategy: all           # ~9.1k
  - name: uitviic
    json_path: __DATA_FOLDER__/uitviic_train.json
    path: uitviic
    modality: image
    sampling_strategy: all           # ~13.5k
  - name: cauldron_vqav2
    json_path: __DATA_FOLDER__/cauldron_vqav2.json
    path: cauldron/vqav2
    modality: image
    sampling_strategy: random:15%    # ~12k of 82.7k
  - name: cauldron_ocrvqa
    json_path: __DATA_FOLDER__/cauldron_ocrvqa.json
    path: cauldron/ocrvqa
    modality: image
    sampling_strategy: random:4%     # ~6.6k of 165.7k
  - name: cauldron_textvqa
    json_path: __DATA_FOLDER__/cauldron_textvqa.json
    path: cauldron/textvqa
    modality: image
    sampling_strategy: random:25%    # ~5.5k of 22k
  - name: viwiki_text
    json_path: __DATA_FOLDER__/viwiki_text.json
    path: ""
    modality: text
    sampling_strategy: first:11000   # ~15% text share, per SmolVLM paper ~14%
```

- [ ] **Step 2: Rewrite `train_1gpu.sh`**

```bash
#!/bin/bash
# Vietnamese SmolVLM2 stage-1 training — 1x GPU (24GB), smolvlm2 stack.
#
# Prerequisites (run once, in order):
#   1. python vision/scripts/data/convert_to_llava_json.py --source ...   (all 5 sources)
#   2. python vision/scripts/tokenizer/build_tokenizer_corpus.py --output ...
#   3. python vision/scripts/tokenizer/expand_tokenizer.py --corpus ... --output_dir $TOKENIZER_DIR
#
# Usage:
#   DATA_FOLDER=/path/to/vietnamese_data TOKENIZER_DIR=/path/to/expanded_tokenizer ./train_1gpu.sh
#   Optional: OUTPUT_DIR=..., MAX_STEPS=20 (smoke test), RESUME=1

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

DATA_FOLDER="${DATA_FOLDER:?Set DATA_FOLDER to the converted-data root}"
TOKENIZER_DIR="${TOKENIZER_DIR:?Set TOKENIZER_DIR to the expanded processor dir}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/checkpoints/vietnamese_stage1}"
MAX_STEPS="${MAX_STEPS:--1}"   # -1 = full epoch; set e.g. 20 for a smoke test

MIXTURE_TEMPLATE="$REPO_ROOT/vision/smolvlm2/scripts/mixtures/vietnamese_stage1.yaml"
mkdir -p "$OUTPUT_DIR"
MIXTURE="$OUTPUT_DIR/mixture_resolved.yaml"
sed "s|__DATA_FOLDER__|$DATA_FOLDER|g" "$MIXTURE_TEMPLATE" > "$MIXTURE"

echo "=== Vietnamese SmolVLM2 stage-1 ==="
echo "data:      $DATA_FOLDER"
echo "tokenizer: $TOKENIZER_DIR"
echo "output:    $OUTPUT_DIR"
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv

cd "$REPO_ROOT/vision/smolvlm2"
export PYTHONPATH="$REPO_ROOT/vision/smolvlm2:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python smolvlm/train/train.py \
    --model_name_or_path HuggingFaceTB/SmolVLM2-500M-Video-Instruct \
    --tokenizer_name_or_path "$TOKENIZER_DIR" \
    --data_mixture "$MIXTURE" \
    --data_folder "$DATA_FOLDER" \
    --output_dir "$OUTPUT_DIR" \
    --num_train_epochs 1 \
    --max_steps "$MAX_STEPS" \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --eval_strategy no \
    --save_strategy steps \
    --save_steps 500 \
    --save_total_limit 2 \
    --learning_rate 1e-4 \
    --weight_decay 0.1 \
    --warmup_steps 100 \
    --lr_scheduler_type cosine \
    --logging_steps 5 \
    --model_max_length 2048 \
    --image_target_size 1536 \
    --gradient_checkpointing True \
    --bf16 True \
    --peft_enable True \
    --lora_rank 16 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --target_modules q_proj k_proj v_proj o_proj \
    --lora_modules_to_save embed_tokens lm_head \
    --report_to none

echo "Done. Checkpoints in: $OUTPUT_DIR"
```

Notes baked into the config: `image_target_size 1536` is the SmolVLM2 default (512 would cripple OCR); LoRA alpha 32 = 2×rank; `embed_tokens lm_head` fully trained because the expanded vocab's new rows must learn (~110M params + ~10M LoRA — fits 24GB with batch 1 + accum 8). If OOM at `model_max_length 2048` + `image_target_size 1536`, first lower `image_target_size` to 1152, then `model_max_length` to 1536.

- [ ] **Step 3: Verify the mixture resolves and parses**

```bash
chmod +x vision/experiments/pretraining/vietnamese/train_1gpu.sh
DATA_FOLDER=/tmp/x TOKENIZER_DIR=/tmp/y OUTPUT_DIR=<scratchpad>/mix_test bash -c '
  sed "s|__DATA_FOLDER__|$DATA_FOLDER|g" vision/smolvlm2/scripts/mixtures/vietnamese_stage1.yaml' \
  | python -c "
import sys, yaml
d = yaml.safe_load(sys.stdin)
assert isinstance(d, dict) and 'vietnamese_stage1' in d
entries = d['vietnamese_stage1']
assert len(entries) == 7
for e in entries:
    assert set(e) >= {'name','json_path','path','modality','sampling_strategy'}, e
    assert '__DATA_FOLDER__' not in e['json_path']
print('mixture OK:', [e['name'] for e in entries])"
```
Expected: `mixture OK: ['viocrvqa', 'openvivqa', 'uitviic', 'cauldron_vqav2', 'cauldron_ocrvqa', 'cauldron_textvqa', 'viwiki_text']`.

- [ ] **Step 4: Commit**

```bash
git add vision/smolvlm2/scripts/mixtures/vietnamese_stage1.yaml \
        vision/experiments/pretraining/vietnamese/train_1gpu.sh
git commit -m "Add Vietnamese stage-1 mixture and rewrite 1-GPU launch script

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 7: Eval metrics module (TDD)

Pure functions → proper pytest cycle. The repo has no test suite; these tests live next to the module and run with plain `pytest`.

**Files:**
- Create: `vision/evaluation/vietnamese/metrics.py`
- Test: `vision/evaluation/vietnamese/test_metrics.py`

**Interfaces:**
- Produces: `normalize_answer(s: str) -> str`, `exact_match(pred: str, refs: list[str]) -> float` (0.0/1.0), `token_f1(pred: str, refs: list[str]) -> float` (max over refs). Consumed by Task 8/9.

- [ ] **Step 1: Write the failing tests**

```python
# vision/evaluation/vietnamese/test_metrics.py
from metrics import exact_match, normalize_answer, token_f1


def test_normalize_strips_case_punctuation_whitespace():
    assert normalize_answer("  Ghi đường Hồ Chí Minh.  ") == "ghi đường hồ chí minh"


def test_normalize_handles_unicode_composition():
    # NFD vs NFC encodings of the same Vietnamese text must normalize equal
    assert normalize_answer("hồ") == normalize_answer("hồ")  # NFC vs NFD source


def test_exact_match_over_multiple_refs():
    assert exact_match("Hà Nội", ["hà nội", "thủ đô"]) == 1.0
    assert exact_match("Sài Gòn", ["hà nội"]) == 0.0


def test_token_f1_partial_overlap():
    # pred shares 3 tokens with a 4-token ref -> p=3/3, r=3/4, f1=6/7
    score = token_f1("có hai người", ["có hai người đàn_ông".replace("_", " ")])
    assert abs(score - 2 * (1.0 * 0.6) / 1.6) < 1e-6 or score > 0.7


def test_token_f1_empty_pred_is_zero():
    assert token_f1("", ["gì đó"]) == 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd vision/evaluation/vietnamese && python -m pytest test_metrics.py -v`
Expected: FAIL/ERROR with `ModuleNotFoundError: No module named 'metrics'`.

- [ ] **Step 3: Implement `metrics.py`**

```python
"""Text metrics for Vietnamese VQA evaluation (exact match, token F1)."""

import re
import string
import unicodedata


def normalize_answer(s: str) -> str:
    """NFC-normalize, lowercase, drop punctuation, collapse whitespace."""
    s = unicodedata.normalize("NFC", s or "")
    s = s.lower()
    s = s.translate(str.maketrans("", "", string.punctuation))
    return re.sub(r"\s+", " ", s).strip()


def exact_match(pred: str, refs: list[str]) -> float:
    p = normalize_answer(pred)
    return 1.0 if any(p == normalize_answer(r) for r in refs) else 0.0


def _f1(pred_tokens: list[str], ref_tokens: list[str]) -> float:
    if not pred_tokens or not ref_tokens:
        return 0.0
    common = {}
    for t in pred_tokens:
        common[t] = common.get(t, 0) + 1
    overlap = 0
    for t in ref_tokens:
        if common.get(t, 0) > 0:
            overlap += 1
            common[t] -= 1
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def token_f1(pred: str, refs: list[str]) -> float:
    """Max token-level F1 of pred against any reference."""
    pred_tokens = normalize_answer(pred).split()
    return max((_f1(pred_tokens, normalize_answer(r).split()) for r in refs),
               default=0.0)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd vision/evaluation/vietnamese && python -m pytest test_metrics.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add vision/evaluation/vietnamese/metrics.py vision/evaluation/vietnamese/test_metrics.py
git commit -m "Add Vietnamese eval metrics (normalize, EM, token-F1) with tests

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 8: Rewrite `tasks.py` as real eval-task loaders

Current `tasks.py` is a placeholder (fake paths, no loading); `run_evaluation.py` imports names that don't exist. Replace `tasks.py` with loaders that read eval splits straight from the HF cache.

**Files:**
- Modify: `vision/evaluation/vietnamese/tasks.py` (full rewrite)

**Interfaces:**
- Consumes: HF cache (same repos as Task 4, but **eval** splits: ViOCRVQA `data/test.json`, OpenViVQA `vlsp2023_dev_data.json` + `dev-images.zip`, UIT-ViIC `test`/`valid`).
- Produces for Task 9:
  - `@dataclass EvalSample: image: PIL.Image.Image; question: str | None; references: list[str]`
  - `@dataclass EvalTask: name: str; kind: str  # "vqa" | "caption"; loader: Callable[[int | None], list[EvalSample]]; prompt: str | None`
  - `TASKS: dict[str, EvalTask]` with keys `viocrvqa_test`, `openvivqa_dev`, `uitviic_test`, `uitviic_valid`.

- [ ] **Step 1: Write the new `tasks.py`**

```python
"""Vietnamese eval tasks: loaders that read eval splits from the HF cache.

Each loader returns a list of EvalSample. VQA samples carry a question and
one or more reference answers; caption samples carry reference captions and
use the task-level prompt.
"""

import io
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from datasets import load_dataset
from huggingface_hub import hf_hub_download
from PIL import Image


@dataclass
class EvalSample:
    image: Image.Image
    question: Optional[str]
    references: list


@dataclass
class EvalTask:
    name: str
    kind: str  # "vqa" | "caption"
    loader: Callable
    prompt: Optional[str] = None  # used when question is None (captioning)


def _zip_index(zf: zipfile.ZipFile) -> dict:
    return {Path(n).name: n for n in zf.namelist()
            if not n.endswith("/") and ".ipynb_checkpoints" not in n}


def load_viocrvqa_test(num_samples=None) -> list:
    zip_path = hf_hub_download("huyhuy123/ViOCRVQA", "data_ViOCRVQA.zip",
                               repo_type="dataset")
    samples = []
    with zipfile.ZipFile(zip_path) as zf:
        data = json.load(zf.open("data/test.json"))
        id_to_file = {img["id"]: img["filename"] for img in data["images"]}
        index = _zip_index(zf)
        for ann in data["annotations"]:
            if num_samples and len(samples) >= num_samples:
                break
            q = (ann.get("question") or "").strip()
            refs = [a.strip() for a in (ann.get("answers") or []) if a and a.strip()]
            member = index.get(id_to_file.get(ann["image_id"], ""))
            if not q or not refs or member is None:
                continue
            image = Image.open(io.BytesIO(zf.read(member))).convert("RGB")
            samples.append(EvalSample(image=image, question=q, references=refs))
    return samples


def load_openvivqa_dev(num_samples=None) -> list:
    json_path = hf_hub_download("uitnlp/OpenViVQA-dataset",
                                "vlsp2023_dev_data.json", repo_type="dataset")
    zip_path = hf_hub_download("uitnlp/OpenViVQA-dataset", "dev-images.zip",
                               repo_type="dataset")
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    id_to_file = {int(k): v for k, v in data["images"].items()}
    samples = []
    with zipfile.ZipFile(zip_path) as zf:
        index = _zip_index(zf)
        for ann in data["annotations"].values():
            if num_samples and len(samples) >= num_samples:
                break
            q = (ann.get("question") or "").strip()
            a = (ann.get("answer") or "").strip()
            member = index.get(id_to_file.get(int(ann["image_id"]), ""))
            if not q or not a or member is None:
                continue
            image = Image.open(io.BytesIO(zf.read(member))).convert("RGB")
            samples.append(EvalSample(image=image, question=q, references=[a]))
    return samples


def _uitviic_from_conversations(split: str, num_samples=None) -> list:
    ds = load_dataset("ThucPD/UIT-ViIC", split=split)
    samples = []
    for row in ds:
        if num_samples and len(samples) >= num_samples:
            break
        try:
            convs = json.loads(row["conversations"])
        except (TypeError, json.JSONDecodeError):
            continue
        caption = next((t["value"].strip() for t in convs if t.get("from") == "gpt"), "")
        if not caption or row.get("image") is None:
            continue
        samples.append(EvalSample(image=row["image"].convert("RGB"),
                                  question=None, references=[caption]))
    return samples


def load_uitviic_test(num_samples=None) -> list:
    return _uitviic_from_conversations("test", num_samples)


def load_uitviic_valid(num_samples=None) -> list:
    """valid split has ~5 human captions per image -> multi-reference scoring."""
    ds = load_dataset("ThucPD/UIT-ViIC", split="valid")
    samples = []
    for row in ds:
        if num_samples and len(samples) >= num_samples:
            break
        caps = row.get("captions")
        if isinstance(caps, str):
            try:
                caps = json.loads(caps.replace("'", '"'))
            except json.JSONDecodeError:
                caps = [caps]
        caps = [c.strip() for c in (caps or []) if c and c.strip()]
        if not caps or row.get("image") is None:
            continue
        samples.append(EvalSample(image=row["image"].convert("RGB"),
                                  question=None, references=caps))
    return samples


CAPTION_PROMPT = "Mô tả bức ảnh này bằng một câu tiếng Việt."

TASKS = {
    "viocrvqa_test": EvalTask("viocrvqa_test", "vqa", load_viocrvqa_test),
    "openvivqa_dev": EvalTask("openvivqa_dev", "vqa", load_openvivqa_dev),
    "uitviic_test": EvalTask("uitviic_test", "caption", load_uitviic_test,
                             prompt=CAPTION_PROMPT),
    "uitviic_valid": EvalTask("uitviic_valid", "caption", load_uitviic_valid,
                              prompt=CAPTION_PROMPT),
}
```

Note for the implementer: `load_uitviic_valid`'s `captions` column may be a Python-repr string (single quotes, seen in exploration) — the `.replace("'", '"')` fallback is a heuristic; if it produces garbage on real data, use `ast.literal_eval` instead. Verify against real rows in Step 2.

- [ ] **Step 2: Smoke-load every task**

```bash
cd vision/evaluation/vietnamese && python - <<'EOF'
from tasks import TASKS
for name, task in TASKS.items():
    samples = task.loader(5)
    assert len(samples) > 0, name
    s = samples[0]
    assert s.image.size[0] > 0 and s.references, name
    if task.kind == "vqa":
        assert s.question, name
    print(f"{name}: {len(samples)} samples; refs[0]={s.references[0][:60]!r}")
EOF
```
Expected: 4 lines with real Vietnamese text in refs, no exceptions.

- [ ] **Step 3: Commit**

```bash
git add vision/evaluation/vietnamese/tasks.py
git commit -m "Rewrite Vietnamese eval tasks as real HF-cache loaders

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 9: Rewrite `run_evaluation.py` + add `compare_results.py`; delete `baseline_eval.py`

**Files:**
- Modify: `vision/evaluation/vietnamese/run_evaluation.py` (full rewrite)
- Create: `vision/evaluation/vietnamese/compare_results.py`
- Delete: `vision/evaluation/vietnamese/baseline_eval.py`

**Interfaces:**
- Consumes: `TASKS`/`EvalTask`/`EvalSample` (Task 8), `exact_match`/`token_f1` (Task 7), optionally a LoRA adapter dir (training output) + expanded processor dir (Task 3).
- Produces: `evals/vietnamese/<run_name>/results.json` with shape `{"model": ..., "adapter": ..., "date": ..., "tasks": {<task>: {"metrics": {...}, "samples": [...]}}}`. `compare_results.py <baseline.json> <finetuned.json>` prints a delta table.
- CLI: `python run_evaluation.py --model_path HuggingFaceTB/SmolVLM2-500M-Video-Instruct [--adapter_path <ckpt>] [--processor_path <expanded_dir>] [--tasks viocrvqa_test,openvivqa_dev,uitviic_valid] [--num_samples 200] [--run_name baseline]`.

- [ ] **Step 1: Write the new `run_evaluation.py`**

```python
"""Unified Vietnamese evaluation runner for SmolVLM2 (base or finetuned).

Baseline:   python run_evaluation.py --run_name baseline
Finetuned:  python run_evaluation.py --run_name finetuned \
                --adapter_path <checkpoint_dir> --processor_path <expanded_processor>
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

import torch
from transformers import AutoProcessor

from metrics import exact_match, token_f1
from tasks import TASKS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_model_and_processor(model_path, adapter_path=None, processor_path=None):
    from transformers import AutoModelForImageTextToText

    processor = AutoProcessor.from_pretrained(processor_path or model_path)
    model = AutoModelForImageTextToText.from_pretrained(
        model_path, torch_dtype=torch.bfloat16
    )
    tokenizer_len = len(processor.tokenizer)
    embed_rows = model.get_input_embeddings().weight.shape[0]
    if tokenizer_len > embed_rows:
        logger.info("Resizing embeddings %d -> %d for expanded tokenizer",
                    embed_rows, tokenizer_len)
        model.resize_token_embeddings(tokenizer_len)
    if adapter_path:
        from peft import PeftModel
        logger.info("Loading LoRA adapter from %s", adapter_path)
        model = PeftModel.from_pretrained(model, adapter_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    return model, processor


@torch.no_grad()
def generate(model, processor, image, text, max_new_tokens):
    messages = [{"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": text},
    ]}]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True,
                                           tokenize=False)
    inputs = processor(text=prompt, images=[image], return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    new_tokens = out[0][inputs["input_ids"].shape[1]:]
    return processor.decode(new_tokens, skip_special_tokens=True).strip()


def evaluate_task(model, processor, task, num_samples, max_new_tokens):
    samples = task.loader(num_samples)
    logger.info("[%s] %d samples", task.name, len(samples))
    records, ems, f1s = [], [], []
    predictions, references = [], []
    for i, s in enumerate(samples):
        text = s.question if s.question else task.prompt
        pred = generate(model, processor, s.image, text, max_new_tokens)
        record = {"i": i, "question": text, "prediction": pred,
                  "references": s.references}
        if task.kind == "vqa":
            record["exact_match"] = exact_match(pred, s.references)
            record["token_f1"] = token_f1(pred, s.references)
            ems.append(record["exact_match"])
            f1s.append(record["token_f1"])
        else:
            predictions.append(pred)
            references.append(s.references)
        records.append(record)
        if (i + 1) % 20 == 0:
            logger.info("[%s] %d/%d", task.name, i + 1, len(samples))

    if task.kind == "vqa":
        metrics = {"exact_match": 100 * sum(ems) / len(ems),
                   "token_f1": 100 * sum(f1s) / len(f1s),
                   "n": len(ems)}
    else:
        import sacrebleu
        from rouge_score import rouge_scorer
        max_refs = max(len(r) for r in references)
        ref_lists = [[r[j] if j < len(r) else r[0] for r in references]
                     for j in range(max_refs)]
        bleu = sacrebleu.corpus_bleu(predictions, ref_lists).score
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
        rl = [max(scorer.score(ref, p)["rougeL"].fmeasure for ref in refs)
              for p, refs in zip(predictions, references)]
        metrics = {"bleu": bleu, "rougeL": 100 * sum(rl) / len(rl),
                   "n": len(predictions)}
    logger.info("[%s] %s", task.name, metrics)
    return {"metrics": metrics, "samples": records}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path",
                        default="HuggingFaceTB/SmolVLM2-500M-Video-Instruct")
    parser.add_argument("--adapter_path", default=None,
                        help="LoRA checkpoint dir (finetuned eval)")
    parser.add_argument("--processor_path", default=None,
                        help="Expanded processor dir (finetuned eval)")
    parser.add_argument("--tasks",
                        default="viocrvqa_test,openvivqa_dev,uitviic_valid")
    parser.add_argument("--num_samples", type=int, default=200)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--run_name", default="baseline")
    parser.add_argument("--output_dir", default=None,
                        help="Default: <repo>/evals/vietnamese/<run_name>")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[3]
    out_dir = Path(args.output_dir) if args.output_dir else (
        repo_root / "evals" / "vietnamese" / args.run_name)
    out_dir.mkdir(parents=True, exist_ok=True)

    model, processor = load_model_and_processor(
        args.model_path, args.adapter_path, args.processor_path)

    results = {"model": args.model_path, "adapter": args.adapter_path,
               "processor": args.processor_path,
               "date": datetime.now().isoformat(), "tasks": {}}
    for name in [t.strip() for t in args.tasks.split(",") if t.strip()]:
        if name not in TASKS:
            raise SystemExit(f"Unknown task {name!r}. Available: {sorted(TASKS)}")
        try:
            results["tasks"][name] = evaluate_task(
                model, processor, TASKS[name], args.num_samples,
                args.max_new_tokens)
        except Exception as e:  # keep the suite going; record the failure
            logger.exception("[%s] failed", name)
            results["tasks"][name] = {"error": f"{type(e).__name__}: {e}"}

    results_file = out_dir / "results.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults -> {results_file}")
    for name, r in results["tasks"].items():
        print(f"  {name}: {r.get('metrics', r.get('error'))}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Write `compare_results.py`**

```python
"""Print a metric delta table between two run_evaluation.py results files.

Usage: python compare_results.py evals/vietnamese/baseline/results.json \
                                 evals/vietnamese/finetuned/results.json
"""

import json
import sys


def main():
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    before = json.load(open(sys.argv[1], encoding="utf-8"))
    after = json.load(open(sys.argv[2], encoding="utf-8"))

    print(f"{'task':<18}{'metric':<14}{'before':>10}{'after':>10}{'delta':>10}")
    print("-" * 62)
    for task in sorted(set(before["tasks"]) | set(after["tasks"])):
        b = before["tasks"].get(task, {}).get("metrics", {})
        a = after["tasks"].get(task, {}).get("metrics", {})
        for metric in sorted(set(b) | set(a)):
            if metric == "n":
                continue
            bv, av = b.get(metric), a.get(metric)
            if bv is None or av is None:
                row = f"{bv if bv is not None else '—':>10}{av if av is not None else '—':>10}{'—':>10}"
            else:
                row = f"{bv:>10.2f}{av:>10.2f}{av - bv:>+10.2f}"
            print(f"{task:<18}{metric:<14}" + row)


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Smoke-run the harness on the base model (tiny sample)**

Run:
```bash
cd vision/evaluation/vietnamese
python run_evaluation.py --run_name smoke --num_samples 3 --max_new_tokens 16 \
    --tasks viocrvqa_test,uitviic_valid
```
Expected: model loads on GPU, both tasks print metrics (values will be poor — that's the point of the baseline), and `evals/vietnamese/smoke/results.json` exists with `tasks.viocrvqa_test.metrics.exact_match` and `tasks.uitviic_valid.metrics.bleu` keys. Then delete the smoke output: `rm -rf <repo>/evals/vietnamese/smoke`.

- [ ] **Step 4: Smoke-test compare_results with the same file twice**

```bash
python run_evaluation.py --run_name smoke --num_samples 2 --max_new_tokens 8 --tasks uitviic_valid
python compare_results.py ../../../evals/vietnamese/smoke/results.json ../../../evals/vietnamese/smoke/results.json
rm -rf ../../../evals/vietnamese/smoke
```
Expected: a table where every delta is `+0.00`.

- [ ] **Step 5: Delete `baseline_eval.py` and commit**

```bash
git rm vision/evaluation/vietnamese/baseline_eval.py
git add vision/evaluation/vietnamese/run_evaluation.py vision/evaluation/vietnamese/compare_results.py
git commit -m "Rewrite eval runner (base/LoRA), add results comparison, drop baseline_eval

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 10: End-to-end smoke test + experiment README

Wire everything together with small samples to prove the full pipeline runs before burning GPU-days, then document the real workflow.

**Files:**
- Create: `vision/experiments/pretraining/vietnamese/README.md`

**Interfaces:**
- Consumes: everything from Tasks 2–9.
- Produces: a verified pipeline + the runbook for the full-scale run.

- [ ] **Step 1: Build a small end-to-end dataset (~200 samples/source)**

```bash
SCRATCH=<scratchpad>/e2e
for src in viocrvqa openvivqa uitviic viwiki; do
  python vision/scripts/data/convert_to_llava_json.py --source $src --output_dir $SCRATCH/data --max_samples 200
done
for sub in vqav2 ocrvqa textvqa; do
  python vision/scripts/data/convert_to_llava_json.py --source cauldron --subset $sub --output_dir $SCRATCH/data --max_samples 200
done
```
Expected: 7 `Wrote ...` lines; `ls $SCRATCH/data` shows all 7 JSONs + image dirs.

- [ ] **Step 2: Build a small corpus and expanded tokenizer**

```bash
python vision/scripts/tokenizer/build_tokenizer_corpus.py \
    --output $SCRATCH/corpus.txt --max_samples 20000 --vqa_repeat 5
python vision/scripts/tokenizer/expand_tokenizer.py \
    --corpus $SCRATCH/corpus.txt --output_dir $SCRATCH/tokenizer --new_vocab_size 51280
```
Expected: all four `OK:` verification lines print; fertility shows a reduction (smaller than the full run's, since only 2k new tokens).

- [ ] **Step 3: 20-step training smoke test**

```bash
DATA_FOLDER=$SCRATCH/data TOKENIZER_DIR=$SCRATCH/tokenizer \
OUTPUT_DIR=$SCRATCH/ckpt MAX_STEPS=20 \
    ./vision/experiments/pretraining/vietnamese/train_1gpu.sh
```
Expected: dataset overview table shows all 7 datasets with nonzero samples; `Resizing embeddings 49280 -> 51280` in the log; `trainable params` printout includes LoRA + embed_tokens + lm_head (~50-120M); loss decreases over 20 steps; a `checkpoint-20` dir appears under `$SCRATCH/ckpt`. If OOM: retry with `MAX_STEPS=20` after lowering `image_target_size` per the note in Task 6.

- [ ] **Step 4: Eval the smoke checkpoint end-to-end**

```bash
cd vision/evaluation/vietnamese
python run_evaluation.py --run_name e2e_base --num_samples 5 --tasks viocrvqa_test
python run_evaluation.py --run_name e2e_ft --num_samples 5 --tasks viocrvqa_test \
    --adapter_path $SCRATCH/ckpt/checkpoint-20 --processor_path $SCRATCH/tokenizer
python compare_results.py ../../../evals/vietnamese/e2e_base/results.json \
                          ../../../evals/vietnamese/e2e_ft/results.json
rm -rf ../../../evals/vietnamese/e2e_base ../../../evals/vietnamese/e2e_ft
```
Expected: both runs complete (20-step metrics will be near-baseline; the point is the adapter+expanded-processor load path works), compare table prints. If `PeftModel.from_pretrained` complains about vocab shape, the adapter was saved before resize — that indicates a Task 5 ordering bug; re-check that `resize_embeddings_for_expanded_tokenizer` runs before `apply_peft`.

- [ ] **Step 5: Write the experiment README**

`vision/experiments/pretraining/vietnamese/README.md` — document the full-scale runbook (exact commands, in order):

```markdown
# Vietnamese SmolVLM2 — stage-1 experiment

Full pipeline (run from repo root; all datasets read from the local HF cache):

## 0. One-time setup
    cd vision/smolvlm2 && pip install -e . && cd -
    pip install rouge_score sacrebleu pyarrow

## 1. Convert training data (~1-2h, dominated by image extraction)
    export DATA_FOLDER=/path/to/vietnamese_data
    for src in viocrvqa openvivqa uitviic viwiki; do
      python vision/scripts/data/convert_to_llava_json.py --source $src --output_dir $DATA_FOLDER
    done
    python vision/scripts/data/convert_to_llava_json.py --source cauldron --subset vqav2  --output_dir $DATA_FOLDER
    python vision/scripts/data/convert_to_llava_json.py --source cauldron --subset ocrvqa --output_dir $DATA_FOLDER
    python vision/scripts/data/convert_to_llava_json.py --source cauldron --subset textvqa --output_dir $DATA_FOLDER

## 2. Expand the tokenizer (49280 -> 57344)
    python vision/scripts/tokenizer/build_tokenizer_corpus.py \
        --output $DATA_FOLDER/corpus.txt --wiki_articles 300000 --vqa_repeat 20
    python vision/scripts/tokenizer/expand_tokenizer.py \
        --corpus $DATA_FOLDER/corpus.txt --output_dir $DATA_FOLDER/tokenizer_vi
    # Check the printed fertility line: expect ~3.5 -> ~1.7-2.0 tokens/word.

## 3. Baseline benchmark (BEFORE training)
    cd vision/evaluation/vietnamese
    python run_evaluation.py --run_name baseline --num_samples 200
    cd -

## 4. Train (1x24GB, ~9.6k steps / 1 epoch)
    TOKENIZER_DIR=$DATA_FOLDER/tokenizer_vi ./vision/experiments/pretraining/vietnamese/train_1gpu.sh
    # Smoke test first: MAX_STEPS=20 TOKENIZER_DIR=... ./train_1gpu.sh

## 5. Benchmark the finetuned model and compare
    cd vision/evaluation/vietnamese
    python run_evaluation.py --run_name finetuned --num_samples 200 \
        --adapter_path <repo>/checkpoints/vietnamese_stage1/checkpoint-<last> \
        --processor_path $DATA_FOLDER/tokenizer_vi
    python compare_results.py ../../../evals/vietnamese/baseline/results.json \
                              ../../../evals/vietnamese/finetuned/results.json

Design spec: docs/superpowers/specs/2026-07-08-vietnamese-smolvlm2-design.md
Mixture: vision/smolvlm2/scripts/mixtures/vietnamese_stage1.yaml (~55% Vi
multimodal / ~30% English cauldron / ~15% Vi text).
```

- [ ] **Step 6: Update CLAUDE.md Vietnamese section**

In the project `CLAUDE.md`, the "Vietnamese SmolVLM experiment" section references the deleted scripts. Rewrite that section's body to:

```markdown
### Vietnamese SmolVLM2 experiment (vision/experiments/pretraining/vietnamese)
Full runbook in `vision/experiments/pretraining/vietnamese/README.md`. Pipeline:
`vision/scripts/data/convert_to_llava_json.py` (HF cache → llava-JSON under
`$DATA_FOLDER`) → `vision/scripts/tokenizer/build_tokenizer_corpus.py` +
`expand_tokenizer.py` (SmolVLM2 vocab 49280→57344) →
`train_1gpu.sh` (wraps `vision/smolvlm2/smolvlm/train/train.py`, mixture
`vision/smolvlm2/scripts/mixtures/vietnamese_stage1.yaml`, LoRA +
trainable embeddings) → `vision/evaluation/vietnamese/run_evaluation.py`
(baseline & finetuned) + `compare_results.py`. Eval outputs land in
`evals/vietnamese/`.
```

Also update the "Running a single eval task / test" bullet for Vietnamese eval to mention `run_evaluation.py --tasks <name>` and `pytest vision/evaluation/vietnamese/test_metrics.py`.

- [ ] **Step 7: Commit**

```bash
git add vision/experiments/pretraining/vietnamese/README.md CLAUDE.md
git commit -m "Add Vietnamese experiment runbook; update CLAUDE.md pipeline docs

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Execution notes

- Tasks 2–4 and 7–8 are independent of each other; Tasks 5–6 depend on 3; Task 9 depends on 7+8; Task 10 depends on all.
- GPU needed only in Task 9 step 3-4 and Task 10 steps 3-4.
- The full-scale run (Task 10's README steps 1–5 with real sizes) is **not** part of this plan's execution — it is the user's GPU-day decision after the pipeline is proven.
