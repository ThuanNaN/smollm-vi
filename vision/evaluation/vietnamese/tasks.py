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
    """Deviation from brief: `data/test.json`'s `answers` field is entirely a
    redacted placeholder -- every one of its 18,601 annotations has
    answers == ["your answer"] (verified across the full file, not a sampling
    artifact). This is a shared-task leaderboard convention (hidden test
    labels); it makes test.json useless for local metric computation. This
    dataset ships a separate `data/dev.json` (4,237 images, 18,587 real,
    diverse answers, zero image-id overlap with train.json or test.json) that
    is a genuine held-out split with real ground truth. We load dev.json here
    instead so the `viocrvqa_test` task actually produces a usable eval set.
    """
    zip_path = hf_hub_download("huyhuy123/ViOCRVQA", "data_ViOCRVQA.zip",
                               repo_type="dataset")
    samples = []
    with zipfile.ZipFile(zip_path) as zf:
        data = json.load(zf.open("data/dev.json"))
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
    """Deviation from brief: the cached `ThucPD/UIT-ViIC` `test` split (like
    `train`, per Task 4's finding) stores `image` as a plain string path (e.g.
    "/dataset/test/images/x.jpg"), not a decoded PIL image -- confirmed
    directly against the cached data (ds.features shows image: Value('string')).
    So `row["image"].convert("RGB")` from the brief would raise
    AttributeError. The real bytes live in this repo's archive.zip under the
    same path with the leading "/" stripped, exactly as
    vision/scripts/data/convert_to_llava_json.py's convert_uitviic handles it.
    """
    ds = load_dataset("ThucPD/UIT-ViIC", split=split)
    zip_path = hf_hub_download("ThucPD/UIT-ViIC", "archive.zip", repo_type="dataset")
    samples = []
    with zipfile.ZipFile(zip_path) as zf:
        members = set(zf.namelist())
        for row in ds:
            if num_samples and len(samples) >= num_samples:
                break
            try:
                convs = json.loads(row["conversations"])
            except (TypeError, json.JSONDecodeError):
                continue
            caption = next((t["value"].strip() for t in convs if t.get("from") == "gpt"), "")
            image_path = row.get("image")
            if not caption or not image_path:
                continue
            member = image_path.lstrip("/")
            if member not in members:
                continue
            image = Image.open(io.BytesIO(zf.read(member))).convert("RGB")
            samples.append(EvalSample(image=image, question=None, references=[caption]))
    return samples


def load_uitviic_test(num_samples=None) -> list:
    return _uitviic_from_conversations("test", num_samples)


def load_uitviic_valid(num_samples=None) -> list:
    """valid split has ~5 human captions per image -> multi-reference scoring.

    Deviation from brief (two points, both verified against the real cached
    repo):

    1. `load_dataset("ThucPD/UIT-ViIC", split="valid")` raises
       `ValueError: Unknown split "valid". Should be one of ['train', 'test']`
       -- the repo's legacy dataset loading script only registers train/test,
       even though a `data/valid-00000-of-00001.parquet` file exists
       alongside train/test parquets. We load that parquet directly via
       `hf_hub_download` + `load_dataset("parquet", ...)` instead.

    2. The brief's `.replace("'", '"')` fallback for a Python-repr `captions`
       string is unnecessary: inspecting the valid parquet's schema and a
       decoded row shows `captions` is already a native Arrow
       `list<string>` column (`datasets` surfaces it as a plain
       `list[str]`), not a stringified repr. Similarly `image` in this split
       is embedded image bytes (an HF `Image` feature, auto-decoded to PIL by
       `datasets` from the parquet's embedded feature metadata), unlike
       train/test's path-into-archive.zip string column. So no
       `ast.literal_eval`/`.replace()` parsing and no zip extraction are
       needed here -- only the train/test splits need the archive.zip
       treatment.
    """
    parquet_path = hf_hub_download("ThucPD/UIT-ViIC",
                                   "data/valid-00000-of-00001.parquet",
                                   repo_type="dataset")
    ds = load_dataset("parquet", data_files={"valid": parquet_path})["valid"]
    samples = []
    for row in ds:
        if num_samples and len(samples) >= num_samples:
            break
        caps = [c.strip() for c in (row.get("captions") or []) if c and c.strip()]
        image = row.get("image")
        if not caps or image is None:
            continue
        samples.append(EvalSample(image=image.convert("RGB"),
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
