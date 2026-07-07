#!/bin/bash
# Vietnamese SmolVLM Stage 1 Training - Single GPU Setup
# For 1x GPU with 24GB VRAM
# Uses the smolvlm2 training infrastructure (HF Trainer based).
#
# Note: the smolvlm2 trainer consumes a data *mixture* yaml
# (see vision/smolvlm2/scripts/mixtures/*.yaml for the format: a list of
# entries with json_path / path / modality / sampling_strategy) plus a
# data_folder root — not raw webdataset shards.

set -e

# Repo root derived from this script's location
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

# ---- User-defined paths (edit these) ----
DATA_MIXTURE="${DATA_MIXTURE:-$REPO_ROOT/vision/smolvlm2/scripts/mixtures/vietnamese_mixture.yaml}"
DATA_FOLDER="${DATA_FOLDER:-/path/to/vietnamese_data}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/checkpoints/vietnamese_stage1_1gpu}"

echo "=========================================="
echo "Vietnamese SmolVLM Stage 1 Training"
echo "=========================================="
echo "GPUs: 1"
echo "Data mixture: $DATA_MIXTURE"
echo "Output dir: $OUTPUT_DIR"
echo "Date: $(date)"
echo ""

# Check GPU availability
echo "Checking GPU..."
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv
echo ""

# Set environment variables
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTHONFAULTHANDLER=1

cd "$REPO_ROOT/vision/smolvlm2"
export PYTHONPATH="$REPO_ROOT/vision/smolvlm2:$PYTHONPATH"

# Run training using smolvlm2
echo "Starting training on single GPU..."
python smolvlm/train/train.py \
    --model_name_or_path HuggingFaceTB/SmolVLM-500M-Instruct \
    --data_mixture "$DATA_MIXTURE" \
    --data_folder "$DATA_FOLDER" \
    --output_dir "$OUTPUT_DIR" \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 2 \
    --learning_rate 0.0001 \
    --weight_decay 0.1 \
    --warmup_steps 100 \
    --lr_scheduler_type "cosine" \
    --logging_steps 5 \
    --model_max_length 512 \
    --image_target_size 512 \
    --gradient_checkpointing True \
    --fp16 False \
    --bf16 True \
    --peft_enable True \
    --lora_rank 16 \
    --lora_alpha 16 \
    --lora_dropout 0.1 \
    --target_modules q_proj v_proj

echo ""
echo "Training complete!"
echo "Check checkpoints in: $OUTPUT_DIR"
