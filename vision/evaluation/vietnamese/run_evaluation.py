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
