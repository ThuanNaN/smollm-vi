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
    # NOTE: deviates from the plan brief. The cached ThucPD/UIT-ViIC parquet
    # stores `image` as a plain string path (e.g.
    # "/dataset/train/images/000000157656.jpg"), not a decoded PIL image as
    # the brief assumed. The actual bytes live in this repo's archive.zip,
    # under the same path with the leading "/" stripped. So we download that
    # zip and extract by exact member path instead of `image.convert(...)`.
    ds = load_dataset("ThucPD/UIT-ViIC", split="train")
    zip_path = hf_hub_download("ThucPD/UIT-ViIC", "archive.zip", repo_type="dataset")
    images_dir = out_root / "uitviic" / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    samples = []
    with zipfile.ZipFile(zip_path) as zf:
        members = set(zf.namelist())
        for row in ds:
            if max_samples and len(samples) >= max_samples:
                break
            try:
                convs = json.loads(row["conversations"])
            except (TypeError, json.JSONDecodeError):
                continue
            if not convs:
                continue
            image_path = row["image"]
            if not image_path:
                continue
            member = image_path.lstrip("/")
            if member not in members:
                continue
            filename = f"{row['id']}.jpg"
            target = images_dir / filename
            if not target.exists():
                with zf.open(member) as src:
                    target.write_bytes(src.read())
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
