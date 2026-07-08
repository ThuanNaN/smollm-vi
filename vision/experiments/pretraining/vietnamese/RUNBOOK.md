# Vietnamese SmolVLM2 stage-1 — portable experiment runbook

Ordered script sequence to run this experiment on a fresh device and produce
a baseline-vs-finetuned results table. The underlying pipeline is already
built and was verified end-to-end via a small smoke test — see
`README.md` in this directory for the pipeline design and the smoke-test
report. This file is the full-scale, portable version of that same
pipeline, meant to be copied to whichever machine actually runs it.

Design spec: `docs/superpowers/specs/2026-07-08-vietnamese-smolvlm2-design.md`

## 0. Environment setup (fresh device)

```bash
git clone <repo-url> smollm-vi && cd smollm-vi
cd vision/smolvlm2
pip install -r requirements.txt      # pins transformers==4.50.0 — do not change, see comment at top of the file
pip install -e . --no-deps
cd -
huggingface-cli login                # needed to pull HuggingFaceTB/SmolVLM2-500M-Video-Instruct + HF datasets
```

If the target GPU has a prebuilt `flash-attn` wheel available for its
torch/CUDA/Python combo, install it and drop `--disable_flash_attn2 True` in
step 4 (faster and lower memory than the `eager` fallback). Otherwise leave
it as-is — `eager` is the verified-working default.

## 1. Download + convert data (~1-2h, dominated by image extraction)

`convert_to_llava_json.py` calls `hf_hub_download`/`load_dataset` itself, so
it will pull each dataset into the local HF cache on first run anyway. If
the target device has slow/flaky network, pre-warm the cache first so a
transient failure during conversion doesn't waste extraction work already
done for earlier sources:

```bash
hf download huyhuy123/ViOCRVQA        --repo-type=dataset
hf download uitnlp/OpenViVQA-dataset  --repo-type=dataset
hf download ThucPD/UIT-ViIC           --repo-type=dataset
hf download vietgpt/wikipedia_vi      --repo-type=dataset
# the_cauldron is a large multi-subset repo (940+ files) — only fetch the
# 3 subsets this pipeline actually uses, via --include:
hf download HuggingFaceM4/the_cauldron --repo-type=dataset --include "vqav2/*"
hf download HuggingFaceM4/the_cauldron --repo-type=dataset --include "ocrvqa/*"
hf download HuggingFaceM4/the_cauldron --repo-type=dataset --include "textvqa/*"
```

Then convert:

```bash
export DATA_FOLDER=/path/to/vietnamese_data
for src in viocrvqa openvivqa uitviic viwiki; do
  python vision/scripts/data/convert_to_llava_json.py --source $src --output_dir $DATA_FOLDER
done
python vision/scripts/data/convert_to_llava_json.py --source cauldron --subset vqav2   --output_dir $DATA_FOLDER
python vision/scripts/data/convert_to_llava_json.py --source cauldron --subset ocrvqa  --output_dir $DATA_FOLDER
python vision/scripts/data/convert_to_llava_json.py --source cauldron --subset textvqa --output_dir $DATA_FOLDER
```

## 2. Expand the tokenizer (49280 -> 57344)

```bash
python vision/scripts/tokenizer/build_tokenizer_corpus.py \
    --output $DATA_FOLDER/corpus.txt --wiki_articles 300000 --vqa_repeat 20
python vision/scripts/tokenizer/expand_tokenizer.py \
    --corpus $DATA_FOLDER/corpus.txt --output_dir $DATA_FOLDER/tokenizer_vi
# Check the printed fertility line: expect ~3.5 -> ~1.7-2.0 tokens/word.
```

## 3. Baseline benchmark (BEFORE training)

Run all 4 eval tasks (not just the 3-task default) so the results table is
complete:

```bash
cd vision/evaluation/vietnamese
python run_evaluation.py --run_name baseline --num_samples 200 \
    --tasks viocrvqa_test,openvivqa_dev,uitviic_valid,uitviic_test
cd -
```

## 4. Train (~9.6k steps / 1 epoch on 1x24GB)

```bash
TOKENIZER_DIR=$DATA_FOLDER/tokenizer_vi \
./vision/experiments/pretraining/vietnamese/train_1gpu.sh
```

- Smoke test first on an unfamiliar environment:
  `MAX_STEPS=20 TOKENIZER_DIR=... ./train_1gpu.sh`
- More VRAM than 24GB: raise `--per_device_train_batch_size` and lower
  `--gradient_accumulation_steps` proportionally (keep effective batch = 8)
  by editing `train_1gpu.sh` directly — faster wall-clock for the same
  optimization trajectory.
- Less VRAM than 24GB: keep batch=1; consider lowering `--image_target_size`
  (currently 1536).
- Time an initial 50-100 steps before committing to the full run to
  extrapolate total wall-clock — the smoke test only ran 20 steps and did
  not measure sustained throughput.

## 5. Benchmark the finetuned model — same tasks as step 3

```bash
cd vision/evaluation/vietnamese
python run_evaluation.py --run_name finetuned --num_samples 200 \
    --tasks viocrvqa_test,openvivqa_dev,uitviic_valid,uitviic_test \
    --adapter_path <repo>/checkpoints/vietnamese_stage1/checkpoint-<last> \
    --processor_path $DATA_FOLDER/tokenizer_vi
```

## 6. Build the comparison table

```bash
python compare_results.py ../../../evals/vietnamese/baseline/results.json \
                          ../../../evals/vietnamese/finetuned/results.json
cd -
```

`compare_results.py` prints a per-task, per-metric delta table
(`exact_match`, `token_f1` for VQA tasks; `bleu`, `rougeL` for the caption
task) — this is the table that demonstrates improvement.

**Keep baseline and finetuned runs comparable**: same `--num_samples`, same
`--tasks`, same `--max_new_tokens` (default 64) in both step 3 and step 5.

## Optional: learning curve across checkpoints

To plot improvement over training instead of a single before/after pair,
repeat step 5 once per saved checkpoint with a distinct `--run_name`
(`train_1gpu.sh` saves every 500 steps, `--save_total_limit 2` by default —
raise that limit first if multiple checkpoints should be kept):

```bash
python run_evaluation.py --run_name finetuned_step500  --adapter_path .../checkpoint-500  ...
python run_evaluation.py --run_name finetuned_step1000 --adapter_path .../checkpoint-1000 ...
```

Then run `compare_results.py` for each pair against the same baseline.

## Open risk to verify before a long run

The last smoke test's environment had `transformers==4.49.0` installed
locally (`torch_env` conda env), one patch version below the
`transformers==4.50.0` pin in `vision/smolvlm2/requirements.txt`. It is not
confirmed whether the smoke test actually ran against 4.49.0 or 4.50.0.
Before committing GPU-hours to the full run on any device, confirm the
installed version matches the pin exactly:

```bash
pip show transformers   # expect 4.50.0
```
