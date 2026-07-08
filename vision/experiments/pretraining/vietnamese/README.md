# Vietnamese SmolVLM2 — stage-1 experiment

Full pipeline (run from repo root; all datasets read from the local HF cache).

## 0. One-time setup

    cd vision/smolvlm2
    uv pip install -r requirements.txt
    uv pip install -e . --no-deps
    cd -

`requirements.txt` pins `transformers==4.50.0` deliberately — see the comment
at the top of that file for why (it's the narrow version window where
`smolvlm` is a known `AutoConfig` model type *and* the pre-refactor
`transformers.models.idefics3` internals this repo's custom
`smolvlm/model/modeling_smolvlm.py` subclasses still exist). Installing with
a newer or older `transformers` will break training with either a `KeyError:
'smolvlm'` (too old) or an `ImportError` on `IDEFICS3_INPUTS_DOCSTRING` (too
new).

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

`train_1gpu.sh` does not pass `--disable_flash_attn2`, so it defaults to
`flash_attention_2`. Unless you separately install `flash-attn` (no prebuilt
wheel for arbitrary torch/CUDA/Python combos on PyPI; compiling from source
is slow), add `--disable_flash_attn2 True` when invoking
`smolvlm/train/train.py` directly, or export it via the script if you patch
it in. This transformers version's `Idefics3VisionTransformer` does not
support `sdpa` for this architecture, so `--disable_flash_attn2 True` falls
back to `eager` attention, which is what the smoke test in this repo actually
used.

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

## Verified end-to-end (smoke test)

The full pipeline above was run end-to-end with small sample sizes (~200
samples/source, 20 training steps, 5 eval samples) to confirm every stage
wires together correctly before committing to a full run:

- Data conversion: all 7 sources converted successfully.
- Tokenizer expansion: 49280 -> 51280 (2000 new tokens for the smoke corpus),
  all verification checks passed, fertility improved.
- Training: embeddings resized 49280 -> 51280, LoRA + `embed_tokens` +
  `lm_head` trainable (~102M params), loss decreased over 20 steps
  (5.46 -> ~5.03), `checkpoint-20` saved.
- Eval: base and finetuned-adapter runs both completed on `viocrvqa_test`;
  `compare_results.py` printed a delta table (`token_f1` +1.99 after 20
  steps — noise-level at this sample size, the point was proving the
  adapter + expanded-processor load path works, not measuring real
  improvement).

Two small, surgical fixes to the pre-existing `vision/smolvlm2` codebase were
required to get training running at all under a `transformers` version new
enough to know the `smolvlm` architecture (see the comment in
`requirements.txt` for the version-window rationale):

- `smolvlm/train/smolvlm_trainer.py`: `ALL_LAYERNORM_LAYERS` moved from
  `transformers.trainer` to `transformers.pytorch_utils` in newer
  `transformers` — import fixed accordingly.
- `smolvlm/model/modeling_smolvlm.py`: `SmolVLMModel.forward()` was missing
  the `cache_position` parameter that upstream `Idefics3Model.forward` now
  accepts and forwards to the text backbone — added it, matching upstream's
  current signature exactly.

`vision/evaluation/vietnamese/run_evaluation.py`'s `generate()` also needed a
one-line fix: explicitly cast `pixel_values` to the model's dtype before
`model.generate(...)`, since this transformers version does not auto-cast it
(a newer version might; don't rely on that).

Neither fix changes training/eval behavior — both restore the codebase's
already-intended behavior against a specific `transformers` version.
