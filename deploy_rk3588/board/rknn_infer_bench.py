"""
在 RK3588 板端运行：加载 .rknn，测推理延迟，并可选采样系统资源。

依赖（板端）:
  pip3 install rknn-toolkit-lite2 numpy

用法:
  python3 rknn_infer_bench.py --rknn student.rknn --warmup 30 --iters 200
  python3 rknn_infer_bench.py --rknn student.rknn --image demo.png --save_mask out.png
"""

from __future__ import annotations

import argparse
import csv
import os
import time
from datetime import datetime

import numpy as np


def read_proc_meminfo():
    out = {}
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as f:
            for line in f:
                parts = line.split(":")
                if len(parts) >= 2:
                    key = parts[0].strip()
                    val = parts[1].strip().split()[0]
                    out[key] = int(val)
    except OSError:
        pass
    return out


def read_npu_load():
    """尝试读取 RK3588 NPU 负载（路径因固件而异）。"""
    candidates = [
        "/sys/kernel/debug/rknpu/load",
        "/sys/kernel/debug/rknpu/freq",
    ]
    info = {}
    for path in candidates:
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    info[path] = f.read().strip()
            except OSError:
                pass
    return info


def sample_bandwidth_hint():
    """
    粗粒度带宽提示：读取 /proc/meminfo 与可选 iostat。
    精确 DDR 带宽需板厂工具（如 rktoolbox / 示波器），此处只记可用指标。
    """
    mem = read_proc_meminfo()
    hint = {
        "MemTotal_kB": mem.get("MemTotal"),
        "MemAvailable_kB": mem.get("MemAvailable"),
        "Cached_kB": mem.get("Cached"),
        "npu": read_npu_load(),
    }
    return hint


def load_image(path: str, size: int, in_ch: int) -> np.ndarray:
    try:
        from PIL import Image
    except ImportError:
        raise SystemExit("板端需要: pip3 install pillow")
    img = Image.open(path).convert("RGB").resize((size, size))
    arr = np.asarray(img, dtype=np.float32)  # HWC
    if in_ch == 1:
        arr = arr.mean(axis=2, keepdims=True)
    # NHWC uint8 风格给 RKNN 更常见；此处 float NCHW 需与 convert 时 mean/std 一致
    # 默认按 toolkit-lite 常用: 直接传 NHWC uint8
    return np.asarray(img.resize((size, size)), dtype=np.uint8)[None, ...]  # NHWC


def main():
    p = argparse.ArgumentParser(description="RK3588 RKNN latency + resource bench")
    p.add_argument("--rknn", type=str, required=True)
    p.add_argument("--input_size", type=int, default=256)
    p.add_argument("--in_channels", type=int, default=3)
    p.add_argument("--warmup", type=int, default=30)
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--core_mask", type=int, default=7, help="NPU core mask, 7=三核")
    p.add_argument("--image", type=str, default="", help="可选真实图；默认随机")
    p.add_argument("--save_mask", type=str, default="")
    p.add_argument("--csv", type=str, default="bench_result.csv")
    p.add_argument("--sample_sys", action="store_true", help="每次推理后采样 meminfo")
    args = p.parse_args()

    try:
        from rknnlite.api import RKNNLite
    except ImportError:
        raise SystemExit("板端安装: pip3 install rknn-toolkit-lite2")

    rknn = RKNNLite()
    print("--> load_rknn")
    ret = rknn.load_rknn(args.rknn)
    if ret != 0:
        raise SystemExit(f"load_rknn failed: {ret}")

    print("--> init_runtime")
    # core_mask: 1/2/4 单核, 3 双核, 7 三核
    ret = rknn.init_runtime(core_mask=args.core_mask)
    if ret != 0:
        raise SystemExit(f"init_runtime failed: {ret}")

    if args.image:
        inp = load_image(args.image, args.input_size, args.in_channels)
    else:
        # 与多数 RKNN demo 一致: NHWC uint8
        inp = np.random.randint(
            0, 256, (1, args.input_size, args.input_size, args.in_channels), dtype=np.uint8
        )

    print(f"[bench] input shape={inp.shape} dtype={inp.dtype}")

    for _ in range(args.warmup):
        rknn.inference(inputs=[inp])

    latencies = []
    sys_samples = []
    for i in range(args.iters):
        t0 = time.perf_counter()
        outputs = rknn.inference(inputs=[inp])
        latencies.append((time.perf_counter() - t0) * 1000.0)
        if args.sample_sys and (i % 20 == 0):
            sys_samples.append(sample_bandwidth_hint())

    arr = np.array(latencies, dtype=np.float64)
    summary = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "rknn": os.path.abspath(args.rknn),
        "rknn_mb": round(os.path.getsize(args.rknn) / 1024 / 1024, 3),
        "input": str(inp.shape),
        "core_mask": args.core_mask,
        "warmup": args.warmup,
        "iters": args.iters,
        "latency_mean_ms": round(float(arr.mean()), 3),
        "latency_std_ms": round(float(arr.std()), 3),
        "latency_p50_ms": round(float(np.percentile(arr, 50)), 3),
        "latency_p90_ms": round(float(np.percentile(arr, 90)), 3),
        "latency_p99_ms": round(float(np.percentile(arr, 99)), 3),
        "fps": round(1000.0 / float(arr.mean()), 2),
    }
    mem = read_proc_meminfo()
    summary["MemAvailable_kB"] = mem.get("MemAvailable")
    summary["MemTotal_kB"] = mem.get("MemTotal")
    npu = read_npu_load()
    summary["npu_load_raw"] = str(npu)[:200]

    print("======== RK3588 Bench ========")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print("================================")

    # 写 CSV（追加）
    write_header = not os.path.isfile(args.csv)
    with open(args.csv, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary.keys()))
        if write_header:
            w.writeheader()
        w.writerow(summary)
    print(f"[save] {os.path.abspath(args.csv)}")

    if args.save_mask and outputs is not None:
        out0 = outputs[0]
        mask = np.array(out0).squeeze()
        # logits -> sigmoid binary
        if mask.dtype != np.uint8:
            mask = 1.0 / (1.0 + np.exp(-mask.astype(np.float32)))
            mask = (mask > 0.5).astype(np.uint8) * 255
        try:
            from PIL import Image
            Image.fromarray(mask.astype(np.uint8)).save(args.save_mask)
            print(f"[save] mask -> {args.save_mask}")
        except Exception as e:
            print(f"[warn] save_mask failed: {e}")

    # 可选：把 latency 明细也存一份
    detail = os.path.splitext(args.csv)[0] + "_latencies.npy"
    np.save(detail, arr)
    print(f"[save] {detail}")

    rknn.release()


if __name__ == "__main__":
    main()
