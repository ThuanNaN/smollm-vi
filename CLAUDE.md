# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository overview

This is Hugging Face's Smol Models repo: training/data/eval code for **SmolLM** (text LLMs) and **SmolVLM** (vision-language models). It is a collection of largely independent pipelines rather than a single application — there is no unified build, lint, or test command. Top-level layout:

```
text/       # SmolLM1/2/3: pretraining configs, finetuning, evaluation, data curation
vision/     # SmolVLM/SmolVLM2: two separate training codebases, data + tokenizer scripts, evaluation
tools/      # Shared local-inference utilities and smol_tools
evals/      # Evaluation run outputs (reports, result JSON) — generated artifacts, not source
```

Each subdirectory generally has its own `README.md` and, where relevant, its own `requirements.txt`. Read the local README before working in a subdirectory — conventions (env setup, expected GPU count, dataset formats) differ per pipeline and are not centralized.

## Commands

There is no repo-wide install/build/test/lint command. Every pipeline below is installed and run independently, in its own environment.

### text/finetuning (TRL + PEFT LoRA finetuning of SmolLM)
```bash
cd text/finetuning
pip install -r requirements.txt
wandb login && huggingface-cli login && accelerate config

accelerate launch train.py \
    --model_id "HuggingFaceTB/SmolLM2-1.7B" \
    --dataset_name "bigcode/the-stack-smol" \
    --subset "data/python" \
    --dataset_text_field "content" \
    --split "train" \
    --max_seq_length 2048 \
    --max_steps 5000 \
    --micro_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --learning_rate 3e-4
```

### text/pretraining (nanotron configs)
These directories (`smollm1/`, `smollm2/`, `smollm3/`, `continual-pretraining/`) contain only nanotron YAML configs, not a runnable script — training itself happens via the external [nanotron](https://github.com/huggingface/nanotron/) repo pointed at these configs.

### text/evaluation (LightEval)
```bash
cd text/evaluation/smollm3  # or smollm2
uv pip install -r requirements.txt

lighteval vllm \
    "model_name=HuggingFaceTB/SmolLM3-3B-Base,dtype=bfloat16,tensor_parallel_size=2" \
    "smollm3_base_test.txt" \
    --custom-tasks "tasks.py" \
    --output-dir "evals/" \
    --save-details
```
Custom eval tasks are defined in each dir's `tasks.py`; the `*.txt` files are task-suite lists passed to `lighteval`.

### vision/m4 (legacy internal VLM training infra, webdataset-based)
```bash
cd vision/m4 && pip install -e .
accelerate launch m4/training/main.py --config <path/to/config.yaml>
```
Evaluation:
```bash
python m4/evaluation/launch.py --commit_hash $(git rev-parse HEAD) \
    --batch_size 16 --mini_batch_size 1024 --do_tasks <TaskName>
```

### vision/smolvlm2 (current HF-Trainer-based VLM training infra)
```bash
cd vision/smolvlm2 && pip install -e .
python smolvlm/train/train.py \
    --model_name_or_path HuggingFaceTB/SmolVLM-256M-Instruct \
    --data_mixture scripts/mixtures/<mixture>.yaml \
    --data_folder <path/to/data> \
    --output_dir checkpoints/<run> \
    --per_device_train_batch_size 4 --learning_rate 1e-5 --num_train_epochs 1
```
For multi-GPU/SLURM, use `scripts/train/multinode*.sh` as templates. DeepSpeed ZeRO configs live in `scripts/zero*.json`.

### Vietnamese SmolVLM experiment (vision/experiments/pretraining/vietnamese)
```bash
cd vision/experiments/pretraining/vietnamese
DATA_FOLDER=/path/to/vietnamese_data ./train_1gpu.sh
```
This wraps `vision/smolvlm2/smolvlm/train/train.py` (not `m4`) — it consumes a data *mixture* YAML (`vision/smolvlm2/scripts/mixtures/*.yaml` format: entries with `json_path`/`path`/`modality`/`sampling_strategy`) plus a `data_folder` root, not raw webdataset shards. Data prep scripts for this track live in `vision/scripts/data/` (Vietnamese Wikipedia extraction, webdataset/OCR-VQA conversion) and `vision/scripts/tokenizer/expand_tokenizer.py` (vocab expansion). Evaluation harness is in `vision/evaluation/vietnamese/`; baseline run outputs land in `evals/vietnamese/`.

### Running a single eval task / test
There is no unit-test suite in this repo. "Testing" here means running a specific eval task:
- LightEval: pass a single task name in the `*.txt` suite file or via `--tasks` to `lighteval`.
- m4 eval: pass one task to `--do_tasks` in `m4/evaluation/launch.py`.
- Vietnamese eval: `python vision/evaluation/vietnamese/run_evaluation.py` (see `tasks.py` for individual task definitions, `baseline_eval.py`/`text_baseline.py` for baseline-only runs).

## Architecture notes

**Two independent VLM training stacks coexist under `vision/`.** `vision/m4` is the original internal codebase (in development since 2022, webdataset-shard input, SLURM-oriented scripts, its own eval harness). `vision/smolvlm2` is the newer HF-`Trainer`-based stack (LoRA/PEFT, DeepSpeed ZeRO, data-mixture-YAML input, image *and* video modalities). New experiment work (e.g. the Vietnamese track) builds on `smolvlm2`, not `m4` — check which stack a script imports from before assuming `m4/training/main.py` is the entry point.

**Text side has three distinct concerns that map to three directories**, each independently versioned per SmolLM generation (`smollm1`/`smollm2`/`smollm3` subfolders):
- `text/pretraining/*`: nanotron config YAMLs only (no code) — actual training runs in the external nanotron repo.
- `text/finetuning/train.py`: the one runnable finetuning script in this repo (TRL `SFTTrainer` + PEFT LoRA); SmolLM2/3 instruct/DPO post-training itself lives externally in the `alignment-handbook` repo, linked from `text/finetuning/README.md`.
- `text/evaluation/*`: LightEval custom task definitions, one dir per model generation.

**`text/data/*` are standalone data-curation pipelines**, not shared library code: `fineweb-edu` trains an educational-content classifier (`run_edu_bert.py`/`train_edu_bert.py`), `finemath`/`smoltalk`/`decontamination` are documented via their own READMEs. Don't assume cross-imports between these.

**`evals/` holds generated evaluation artifacts** (markdown reports, result JSON), not source — treat it like a build output directory when deciding what to read for behavior vs. what to read for historical results.

**No CI beyond secret scanning.** `.github/workflows/trufflehog.yml` runs TruffleHog on every push; there is no lint/type-check/test workflow, so don't assume `git push` will be gated by anything beyond that.
