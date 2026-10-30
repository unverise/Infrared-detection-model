"""
PC 侧 ONNXRuntime 基准：与板端结果对照用（非 NPU 速度）。

用法:
  python bench_onnx_pc.py --onnx outputs/student.onnx --warmup 20 --iters 100
"""

from __future__ import annotations

import argparse
import time

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--onnx", type=str, required=True)
    p.add_argument("--input_size", type=int, default=256)
    p.add_argument("--in_channels", type=int, default=3)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--iters", type=int, default=100)
    args = p.parse_args()

    try:
        import onnxruntime as ort
    except ImportError:
        raise SystemExit("pip install onnxruntime")

    sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    name = inp.name
    x = np.random.randn(1, args.in_channels, args.input_size, args.input_size).astype(np.float32)

    for _ in range(args.warmup):
        sess.run(None, {name: x})

    t0 = time.perf_counter()
    for _ in range(args.iters):
        sess.run(None, {name: x})
    dt = (time.perf_counter() - t0) / args.iters * 1000.0
    print(f"[PC ORT] latency={dt:.2f} ms/img  (~{1000.0/dt:.1f} FPS)  input={x.shape}")


if __name__ == "__main__":
    main()
