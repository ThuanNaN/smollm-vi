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
#   Optional: DISABLE_FLASH_ATTN2=False if flash-attn is installed (faster,
#     lower memory than the eager default — see vision/smolvlm2/requirements.txt
#     for why flash-attn isn't installed by default).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

DATA_FOLDER="${DATA_FOLDER:?Set DATA_FOLDER to the converted-data root}"
TOKENIZER_DIR="${TOKENIZER_DIR:?Set TOKENIZER_DIR to the expanded processor dir}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/checkpoints/vietnamese_stage1}"
MAX_STEPS="${MAX_STEPS:--1}"   # -1 = full epoch; set e.g. 20 for a smoke test
DISABLE_FLASH_ATTN2="${DISABLE_FLASH_ATTN2:-True}"   # True = eager attention (no flash-attn needed)

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
    --disable_flash_attn2 "$DISABLE_FLASH_ATTN2" \
    --report_to none

echo "Done. Checkpoints in: $OUTPUT_DIR"
