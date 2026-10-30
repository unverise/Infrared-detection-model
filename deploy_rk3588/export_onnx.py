"""
PC: 将 DNANet 教师/学生权重导出为单输出 ONNX，供 RK3588 转换。

用法（在 deploy_rk3588 目录）:
  python export_onnx.py --ckpt path/to/student_best.pth.tar --out outputs/student.onnx
"""

from __future__ import annotations

import argparse
import os
import sys

import torch
import torch.nn as nn

# 保证能 import Infrared-Small-Target-Detection-master 下的 model/
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_DNANET = os.path.join(_ROOT, "Infrared-Small-Target-Detection-master")
if _DNANET not in sys.path:
    sys.path.insert(0, _DNANET)

from model.load_param_data import load_param  # noqa: E402
from model.model_DNANet import DNANet, Res_CBAM_block  # noqa: E402


class SingleOutputWrapper(nn.Module):
    """deep supervision 时取最后一层，保证 ONNX/RKNN 单输出。"""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x):
        out = self.model(x)
        if isinstance(out, (list, tuple)):
            return out[-1]
        return out


def load_ckpt(path: str):
    ckpt = torch.load(path, map_location="cpu")
    if not isinstance(ckpt, dict):
        raise ValueError(f"unexpected ckpt type: {type(ckpt)}")
    state = ckpt.get("state_dict", ckpt)
    if not isinstance(state, dict):
        raise ValueError(f"no state_dict in {path}")
    state = {k.replace("module.", ""): v for k, v in state.items()}
    nb_filter = ckpt.get("nb_filter", None)
    # 学生 ckpt 通常不写 deep_supervision；有 final1 则按 DS 构图
    deep_sup = ckpt.get("deep_supervision", None)
    if deep_sup is None:
        deep_sup = any(k.startswith("final1") for k in state) or any(
            k.startswith("final4") for k in state
        )
    num_blocks = ckpt.get("num_blocks", None)
    return state, nb_filter, deep_sup, num_blocks, ckpt


def build_model(args, nb_filter, num_blocks, deep_sup):
    if nb_filter is None:
        nb_filter, num_blocks_ld = load_param(args.channel_size, args.backbone)
        if num_blocks is None:
            num_blocks = num_blocks_ld
    if num_blocks is None:
        _, num_blocks = load_param(args.channel_size, args.backbone)
    model = DNANet(
        num_classes=1,
        input_channels=args.in_channels,
        block=Res_CBAM_block,
        num_blocks=list(num_blocks),
        nb_filter=list(nb_filter),
        deep_supervision=deep_sup,
    )
    return model, nb_filter, num_blocks


def main():
    p = argparse.ArgumentParser(description="Export DNANet to ONNX for RK3588")
    p.add_argument("--ckpt", type=str, required=True, help="teacher or student .pth.tar")
    p.add_argument("--out", type=str, default="outputs/dnanet.onnx")
    p.add_argument("--input_size", type=int, default=256)
    p.add_argument("--in_channels", type=int, default=3)
    p.add_argument("--channel_size", type=str, default="three")
    p.add_argument("--backbone", type=str, default="resnet_18")
    p.add_argument("--opset", type=int, default=11, help="RKNN 常用 11/12")
    p.add_argument("--nb_filter", type=str, default="",
                   help="可选: 手动指定学生通道, 如 13,22,38,64,102")
    args = p.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)

    state, nb_filter, deep_sup, num_blocks, meta = load_ckpt(args.ckpt)
    if args.nb_filter.strip():
        nb_filter = [int(x) for x in args.nb_filter.split(",")]
    if nb_filter is None:
        print("[warn] ckpt 无 nb_filter，按教师默认 channel_size 构建；学生权重请带 nb_filter")
    model, nb_filter, num_blocks = build_model(args, nb_filter, num_blocks, deep_sup)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print("[warn] missing keys:", missing[:6], "..." if len(missing) > 6 else "")
    if unexpected:
        print("[warn] unexpected keys:", unexpected[:6], "..." if len(unexpected) > 6 else "")

    wrapped = SingleOutputWrapper(model).eval()
    dummy = torch.randn(1, args.in_channels, args.input_size, args.input_size)

    with torch.no_grad():
        y = wrapped(dummy)
    print(f"[export] nb_filter={list(nb_filter)} num_blocks={list(num_blocks)}")
    print(f"[export] input={tuple(dummy.shape)} output={tuple(y.shape)}")
    if "best_iou" in meta:
        print(f"[export] ckpt best_iou={meta['best_iou']}")

    torch.onnx.export(
        wrapped,
        dummy,
        args.out,
        input_names=["input"],
        output_names=["mask_logits"],
        opset_version=args.opset,
        do_constant_folding=True,
        dynamic_axes=None,  # RKNN 固定 shape 更稳
    )
    print(f"[save] ONNX -> {os.path.abspath(args.out)}")
    print(f"[size] {os.path.getsize(args.out) / 1024 / 1024:.2f} MB")


if __name__ == "__main__":
    main()
