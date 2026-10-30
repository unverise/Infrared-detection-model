#!/usr/bin/env bash
# 在 tmux 里顺序全量训练 4 个数据集上的 UIU-Net
# 用法: bash run_uiunet_all.sh
set -euo pipefail
cd "$(dirname "$0")"

DATASET_DIR="${DATASET_DIR:-../RDANet-main/datasets}"
EPOCHS="${EPOCHS:-500}"
BS="${BS:-3}"
GPUS="${GPUS:-0}"

mkdir -p logs

for ds in NUAA-SIRST NUDT-SIRST IRSTD-1k SIRST-V2; do
  echo "======== start $ds $(date) ========"
  python train_irstd.py \
    --dataset "$ds" \
    --dataset_dir "$DATASET_DIR" \
    --epochs "$EPOCHS" \
    --batch_size "$BS" \
    --gpus "$GPUS" \
    --eval_every 10 \
    2>&1 | tee "logs/uiunet_${ds}_$(date +%Y%m%d_%H%M%S).log"
  echo "======== done $ds $(date) ========"
done

echo "ALL UIU-Net jobs finished."
