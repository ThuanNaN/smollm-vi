#!/bin/bash
# Vietnamese SmolVLM2 stage-1 training — 3x GPU (24GB each, e.g. RTX A5000), smolvlm2 stack.
#
# Multi-GPU sibling of train_1gpu.sh. Uses torchrun -> DistributedDataParallel
# (one process per GPU), NOT the legacy nn.DataParallel that a plain `python`
# launch silently falls into. Memory efficiency on 24GB cards comes from:
#   * DDP + gradient checkpointing (use_reentrant=False) — activations, the
#     dominant cost for 1536px image batches, are recomputed instead of stored.
#   * bf16 weights + tf32 matmuls (Ampere+).
#   * LoRA + trainable embeddings only — optimizer state stays small, so
#     DeepSpeed ZeRO sharding buys little here and is deliberately not used.
#
# Prerequisites: identical to train_1gpu.sh (convert data, build+expand tokenizer).
#
# Usage:
#   DATA_FOLDER=/path/to/vietnamese_data TOKENIZER_DIR=/path/to/expanded_tokenizer ./train_gpus.sh
#   Optional: OUTPUT_DIR=..., MAX_STEPS=20 (smoke test), NUM_GPUS=3,
#             PER_DEVICE_BATCH=1, GRAD_ACCUM=3, DISABLE_FLASH_ATTN2=False
#   Pick specific cards with e.g. CUDA_VISIBLE_DEVICES=0,1,2 ./train_gpus.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

DATA_FOLDER="${DATA_FOLDER:?Set DATA_FOLDER to the converted-data root}"
TOKENIZER_DIR="${TOKENIZER_DIR:?Set TOKENIZER_DIR to the expanded processor dir}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/checkpoints/vietnamese_stage1_3gpu}"
MAX_STEPS="${MAX_STEPS:--1}"   # -1 = full epoch; set e.g. 20 for a smoke test
DISABLE_FLASH_ATTN2="${DISABLE_FLASH_ATTN2:-True}"   # True = eager attention (no flash-attn needed)

NUM_GPUS="${NUM_GPUS:-3}"
PER_DEVICE_BATCH="${PER_DEVICE_BATCH:-4}"
# Global batch = PER_DEVICE_BATCH * NUM_GPUS * GRAD_ACCUM.
# Default 1*3*3 = 9, matching the 1-GPU script's effective batch of 8 (1*8).
GRAD_ACCUM="${GRAD_ACCUM:-3}"

# Resolve to absolute paths: torchrun launches train.py with CWD=vision/smolvlm2
# below, so any relative DATA_FOLDER/TOKENIZER_DIR would resolve against the wrong dir.
DATA_FOLDER="$(realpath "$DATA_FOLDER")"
TOKENIZER_DIR="$(realpath "$TOKENIZER_DIR")"
OUTPUT_DIR="$(realpath -m "$OUTPUT_DIR")"   # -m: may not exist yet (created below)

MIXTURE_TEMPLATE="$REPO_ROOT/vision/smolvlm2/scripts/mixtures/vietnamese_stage1.yaml"
mkdir -p "$OUTPUT_DIR"
MIXTURE="$OUTPUT_DIR/mixture_resolved.yaml"
sed "s|__DATA_FOLDER__|$DATA_FOLDER|g" "$MIXTURE_TEMPLATE" > "$MIXTURE"

echo "=== Vietnamese SmolVLM2 stage-1 (DDP x${NUM_GPUS}) ==="
echo "data:        $DATA_FOLDER"
echo "tokenizer:   $TOKENIZER_DIR"
echo "output:      $OUTPUT_DIR"
echo "global batch: ${PER_DEVICE_BATCH} x ${NUM_GPUS} x ${GRAD_ACCUM} = $((PER_DEVICE_BATCH * NUM_GPUS * GRAD_ACCUM))"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv

cd "$REPO_ROOT/vision/smolvlm2"
export PYTHONPATH="$REPO_ROOT/vision/smolvlm2:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# NCCL's shared-memory transport corrupts the /dev/shm segment name on this
# multi-NUMA PCIe topology (the rank on the cross-NUMA "SYS" GPU crashes on the
# first collective with a UnicodeDecodeError on byte 0x95). Disabling SHM routes
# intra-node comms over PCIe P2P / socket instead — verified working. Set
# NCCL_SHM_DISABLE=0 to re-enable if a future NCCL fixes the bug.
export NCCL_SHM_DISABLE="${NCCL_SHM_DISABLE:-1}"

# DataLoader workers fork after the fast tokenizer has been used; silence the
# tokenizers fork-parallelism warning by choosing the fork-safe setting explicitly.
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

torchrun --standalone --nproc_per_node="$NUM_GPUS" \
    smolvlm/train/train.py \
    --model_name_or_path HuggingFaceTB/SmolVLM2-500M-Video-Instruct \
    --tokenizer_name_or_path "$TOKENIZER_DIR" \
    --data_mixture "$MIXTURE" \
    --data_folder "$DATA_FOLDER" \
    --output_dir "$OUTPUT_DIR" \
    --num_train_epochs 1 \
    --max_steps "$MAX_STEPS" \
    --per_device_train_batch_size "$PER_DEVICE_BATCH" \
    --gradient_accumulation_steps "$GRAD_ACCUM" \
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
    --tf32 True \
    --dataloader_num_workers 4 \
    --dataloader_drop_last True \
    --ddp_find_unused_parameters False \
    --peft_enable True \
    --lora_rank 16 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --target_modules q_proj k_proj v_proj o_proj \
    --lora_modules_to_save embed_tokens lm_head \
    --disable_flash_attn2 "$DISABLE_FLASH_ATTN2" \
    --report_to wandb \
    --run_name vietnamese_stage1_3gpu

echo "Done. Checkpoints in: $OUTPUT_DIR"
