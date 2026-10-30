#!/usr/bin/env bash
# 在 RK3588 板端与推理并行采集系统资源（内存 / NPU / 粗粒度带宽相关）
# 用法: bash collect_system_stats.sh [interval_sec] [out_csv]
# 例:   bash collect_system_stats.sh 0.5 sys_stats.csv &
#       python3 rknn_infer_bench.py --rknn student.rknn
#       kill %1

set -euo pipefail
INTERVAL="${1:-0.5}"
OUT="${2:-sys_stats.csv}"

echo "timestamp,MemTotal_kB,MemAvailable_kB,MemFree_kB,Cached_kB,Buffers_kB,npu_raw" > "$OUT"

read_mem() {
  local key="$1"
  awk -v k="$key" '$1==k":" {print $2}' /proc/meminfo 2>/dev/null || echo ""
}

read_npu() {
  if [[ -r /sys/kernel/debug/rknpu/load ]]; then
    tr '\n' ' ' < /sys/kernel/debug/rknpu/load | tr ',' ';'
  else
    echo "na"
  fi
}

echo "[collect] interval=${INTERVAL}s -> ${OUT}  (Ctrl+C 停止)"
while true; do
  ts="$(date +%Y-%m-%dT%H:%M:%S)"
  mt="$(read_mem MemTotal)"
  ma="$(read_mem MemAvailable)"
  mf="$(read_mem MemFree)"
  ca="$(read_mem Cached)"
  bu="$(read_mem Buffers)"
  npu="$(read_npu)"
  echo "${ts},${mt},${ma},${mf},${ca},${bu},\"${npu}\"" >> "$OUT"
  sleep "$INTERVAL"
done
