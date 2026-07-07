# Vietnamese SmolVLM2 Adaptation — Design

Date: 2026-07-08. Status: approved by user.

## Goal

Adapt SmolVLM2-500M-Video-Instruct to Vietnamese via (1) tokenizer vocabulary
expansion, (2) stage-1 image training (LoRA + trainable embeddings) on a
Vietnamese + English data mixture, (3) before/after benchmarking on Vietnamese
VQA/captioning tasks. Hardware: 1× GPU 24GB. Training stack: `vision/smolvlm2`
(HF Trainer). The half-finished m4-track files are deleted.

## Decisions (locked)

- **Stack**: smolvlm2 only. Delete m4-track files (m4 config yaml, accelerate
  config, ds_config, webdataset converters, m4 pyproject; revert m4/__init__.py).
- **Base model**: `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` (vocab 49280,
  hidden 960, untied embeddings, `<image>` = 49190).
- **Tokenizer**: expand 49280 → 57344 (+8064, multiple of 128).
- **Training**: single stage-1 image run. LoRA on attention projections +
  full-train `embed_tokens` and `lm_head` via `modules_to_save`. New embedding
  rows mean-initialized from base sub-token embeddings.
- **Measured motivation**: Vietnamese fertility on the base tokenizer is 3.51
  tokens/word vs 1.13 for English (wikipedia_vi sample, 200 articles).

## 1. Data preparation

All sources already in HF cache. New script
`vision/scripts/data/convert_to_llava_json.py` converts each source to
llava-style JSON + images on disk under `$DATA_FOLDER`:

```
$DATA_FOLDER/
  viocrvqa/images/*.jpg        viocrvqa_train.json
  openvivqa/images/*.jpg       openvivqa_train.json
  uitviic/images/*.jpg         uitviic_train.json
  cauldron/<subset>/images/    cauldron_<subset>.json
  viwiki_text.json             (text-only)
```

Sample format: `{"image": "images/x.jpg", "conversations": [{"from": "human",
"value": "<image>\n..."}, {"from": "gpt", "value": "..."}]}`. Text-only samples
omit `image`.

Source specifics (train splits only; eval splits reserved):

| Source | Train size | Handling |
|---|---|---|
| ViOCRVQA (zip → COCO-style json, `bf_all_img/`) | 86,592 QA / 19,775 imgs | group QA per image into one multi-turn conversation; take first of `answers[]`; skip `.ipynb_checkpoints`/corrupt images |
| OpenViVQA (vlsp2023_*.json + image zips) | 30,833 QA / 9,129 imgs | same multi-turn grouping |
| UIT-ViIC (parquet, already llava-format) | 13,481 | write image bytes to disk, pass conversations through |
| the_cauldron subsets: vqav2, ocrvqa, textvqa, textcaps, cocoqa | 82.7k / 165.7k / 22k / 22k / 46k | parquet `images[]`+`texts[]` → llava json (English retention) |
| wikipedia_vi | 1.28M articles | ~30–50k articles, trimmed ~1500 chars, wrapped as text-only conversations |

Eval splits reserved: ViOCRVQA test (18,601 QA), OpenViVQA dev (3,545 QA; test
is multi-answer/competition-style — not used), UIT-ViIC test (1,155) and valid
(924, ~5 refs/image for multi-reference BLEU).

## 2. Mixture (`vision/smolvlm2/scripts/mixtures/vietnamese_stage1.yaml`)

| Group | Sources | Effective samples | Share |
|---|---|---|---|
| Vietnamese multimodal | viocrvqa 100%, openvivqa 100%, uitviic 100% | ~42k | ~55% |
| English multimodal | vqav2 random:15%, ocrvqa random:4%, textvqa random:25% | ~24k | ~30% |
| Vietnamese text | viwiki ~11k | ~11k | ~15% |

Total ~77k samples ≈ 1 epoch ≈ ~9.6k optimizer steps at effective batch 8.
Proportions are a starting point per the SmolVLM paper (~14% text); adjust
after benchmarking.

## 3. Tokenizer expansion

- New script `vision/scripts/tokenizer/build_tokenizer_corpus.py`: corpus =
  wikipedia_vi ~300k articles (~800MB) + train-split text of
  ViOCRVQA/OpenViVQA/UIT-ViIC (10MB) repeated ×20 (≈15–20% of corpus). Train
  splits only. Args: `--wiki_articles 300000 --vqa_repeat 20`. Replaces
  `extract_vietnamese_wikipedia.py`.
- `expand_tokenizer.py` (existing logic kept: train byte-level BPE on corpus,
  append unseen tokens+merges to base tokenizer.json, base IDs unchanged) with
  changes: base = SmolVLM2-500M-Video-Instruct **processor** (save full
  processor dir so `AutoProcessor.from_pretrained(<dir>)` works); default
  target 57344; built-in verification:
  - `<image>` still 49190; all special tokens keep IDs
  - Vietnamese encode/decode roundtrip lossless
  - fertility before/after on held-out Vietnamese text (print numbers)
  - English sample IDs identical to base (warn if not)

## 4. smolvlm2 trainer changes (minimal)

- Fix pre-existing bug: `train.py:323` calls `apply_peft_if_needed` but the
  function is named `apply_peft` (crashes with `--peft_enable True`).
- New args: `--tokenizer_name_or_path` (load AutoProcessor from expanded dir
  when set), `--lora_modules_to_save` (default empty; we pass
  `embed_tokens lm_head`).
- After model load: if `len(tokenizer) > vocab_size` →
  `resize_token_embeddings` + mean-init new rows of both `embed_tokens` and
  `lm_head` (untied) from base sub-token embeddings.

## 5. Training config (`train_1gpu.sh`, rewritten)

- model `HuggingFaceTB/SmolVLM2-500M-Video-Instruct`, tokenizer = expanded dir
- mixture `vietnamese_stage1.yaml`
- LoRA: rank 16, alpha 32, dropout 0.1, target `q_proj k_proj v_proj o_proj`,
  modules_to_save `embed_tokens lm_head` (~110M trainable + ~10M LoRA)
- `model_max_length 2048`, `image_target_size 1536` (default; 512 was too
  small and would hurt OCR), bf16, gradient checkpointing, effective batch 8
  (per-device 1–2 × grad accum), lr 1e-4 cosine, warmup 100

## 6. Evaluation harness (`vision/evaluation/vietnamese/`)

Rewrite `tasks.py` + `run_evaluation.py` into a working harness (current
run_evaluation.py imports names that don't exist; tasks.py is a placeholder):

- Tasks: ViOCRVQA-test, OpenViVQA-dev, UIT-ViIC-test/valid
- Metrics: exact match + token-F1 (VQA), BLEU/ROUGE-L (captioning,
  multi-reference on valid split)
- Model loading: base model OR checkpoint with LoRA adapter + expanded
  processor (PeftModel), bf16, greedy decode
- Results to `evals/vietnamese/<run_name>/results.json`; a compare script
  prints a before/after delta table
- Fold `baseline_eval.py` into this harness (delete it); keep
  `text_baseline.py` as-is

Workflow: run harness on base model (baseline) → train → run harness on
finetuned checkpoint → compare.

## 7. Cleanup (delete)

- `vision/experiments/pretraining/vietnamese/config_stage1_1gpu.yaml`,
  `accelerate_config_1gpu.yaml`, `ds_config_zero2.json` (m4-style, unused)
- `vision/scripts/data/convert_vietnamese_to_webdataset.py`,
  `convert_viocrvqa.py` (webdataset/m4 format)
- `vision/m4/pyproject.toml`; revert `vision/m4/__init__.py`
- `vision/evaluation/vietnamese/baseline_eval.py` (absorbed into harness)
- `vision/scripts/data/extract_vietnamese_wikipedia.py` (replaced by
  `build_tokenizer_corpus.py`)

## 8. Verification strategy

Every script runs against real cache data with `--max_samples` small before
full runs; training smoke-tested ~20 steps before the real run. Known risk:
after expansion the checkpoint requires the expanded processor — always ship
them together.
