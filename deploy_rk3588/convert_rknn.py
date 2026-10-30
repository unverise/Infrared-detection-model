"""
PC: ONNX -> RKNN（rk3588）。

依赖（建议独立 conda）:
  pip install rknn-toolkit2

用法:
  python convert_rknn.py --onnx outputs/student.onnx --out outputs/student.rknn
  python convert_rknn.py --onnx outputs/student.onnx --quantize --dataset configs/calib_list.txt
"""

from __future__ import annotations

import argparse
import os
import sys


def main():
    p = argparse.ArgumentParser(description="Convert ONNX to RKNN for RK3588")
    p.add_argument("--onnx", type=str, required=True)
    p.add_argument("--out", type=str, default="")
    p.add_argument("--target", type=str, default="rk3588")
    p.add_argument("--quantize", action="store_true", help="INT8 量化（需校准图列表）")
    p.add_argument("--dataset", type=str, default="configs/calib_list.txt",
                   help="每行一张校准图片路径（与 ONNX 输入分辨率一致）")
    p.add_argument("--mean", type=float, nargs=3, default=[0.0, 0.0, 0.0])
    p.add_argument("--std", type=float, nargs=3, default=[255.0, 255.0, 255.0],
                   help="与训练归一化对齐；默认按 uint8/255 近似")
    args = p.parse_args()

    if not os.path.isfile(args.onnx):
        print(f"[error] ONNX not found: {args.onnx}")
        sys.exit(1)

    out = args.out or (os.path.splitext(args.onnx)[0] + ".rknn")
    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)

    try:
        from rknn.api import RKNN
    except ImportError:
        print("[error] 未安装 rknn-toolkit2。请在 PC 上: pip install rknn-toolkit2")
        sys.exit(1)

    rknn = RKNN(verbose=True)
    print("--> config")
    rknn.config(
        mean_values=[list(args.mean)],
        std_values=[list(args.std)],
        target_platform=args.target,
    )

    print("--> load_onnx")
    ret = rknn.load_onnx(model=args.onnx)
    if ret != 0:
        print("load_onnx failed")
        sys.exit(ret)

    print("--> build")
    do_quant = bool(args.quantize)
    if do_quant and not os.path.isfile(args.dataset):
        print(f"[error] quantize 需要 dataset 列表文件: {args.dataset}")
        sys.exit(1)
    ret = rknn.build(do_quantization=do_quant, dataset=args.dataset if do_quant else None)
    if ret != 0:
        print("build failed")
        sys.exit(ret)

    print("--> export_rknn")
    ret = rknn.export_rknn(out)
    if ret != 0:
        print("export_rknn failed")
        sys.exit(ret)

    rknn.release()
    print(f"[save] RKNN -> {os.path.abspath(out)}")
    print(f"[size] {os.path.getsize(out) / 1024 / 1024:.2f} MB")


if __name__ == "__main__":
    main()
