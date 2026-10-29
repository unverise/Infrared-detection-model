from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn

from compress.adapters.dnanet_adapter import DNA_NET_IN_LAYOUT, DNANetAdapter
from model.model_DNANet import DNANet, Res_CBAM_block, VGG_CBAM_Block


def save_keep_masks(path: str, keep_idx: Dict[int, torch.Tensor], meta: dict):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {"keep_idx": {int(k): v.cpu() for k, v in keep_idx.items()}, "meta": meta}
    torch.save(payload, path)
    print(f"[save] keep masks -> {path}")


def load_keep_masks(path: str):
    payload = torch.load(path, map_location="cpu")
    keep_idx = {int(k): v for k, v in payload["keep_idx"].items()}
    return keep_idx, payload.get("meta", {})


def _slice_conv(src: nn.Conv2d, dst: nn.Conv2d, keep_out: Optional[torch.Tensor], keep_in: Optional[torch.Tensor]):
    with torch.no_grad():
        w = src.weight.data
        if keep_out is not None:
            w = w[keep_out]
        if keep_in is not None:
            w = w[:, keep_in]
        dst.weight.data.copy_(w)
        if src.bias is not None and dst.bias is not None:
            b = src.bias.data
            if keep_out is not None:
                b = b[keep_out]
            dst.bias.data.copy_(b)


def _slice_bn(src: nn.BatchNorm2d, dst: nn.BatchNorm2d, keep_out: torch.Tensor):
    with torch.no_grad():
        dst.weight.data.copy_(src.weight.data[keep_out])
        dst.bias.data.copy_(src.bias.data[keep_out])
        dst.running_mean.copy_(src.running_mean.data[keep_out])
        dst.running_var.copy_(src.running_var.data[keep_out])
        dst.num_batches_tracked.copy_(src.num_batches_tracked)


def _slice_ca(src, dst, keep_out: torch.Tensor):
    # ChannelAttention: fc1 out = in_planes//16, fc2 out = in_planes
    # When channels change, rebuild already has new dims; copy overlapping mid dims if divisible.
    with torch.no_grad():
        # simplest robust path: re-init CA on student (already random), or
        # slice if ratios match.
        sin, din = src.fc1.in_channels, dst.fc1.in_channels
        if sin == keep_out.numel() and din == keep_out.numel():
            # teacher already matched keep size? then direct copy
            dst.fc1.weight.data.copy_(src.fc1.weight.data)
            dst.fc2.weight.data.copy_(src.fc2.weight.data)
            return
        # teacher full -> student kept
        # fc1: [C/16, C, 1, 1] -> need new C'/16 and C'
        # Approximate: slice input channels; re-init reduced mid if shapes differ.
        try:
            src_mid = src.fc1.out_channels
            dst_mid = dst.fc1.out_channels
            mid_keep = torch.arange(min(src_mid, dst_mid))
            w1 = src.fc1.weight.data[mid_keep][:, keep_out]
            if w1.shape == dst.fc1.weight.shape:
                dst.fc1.weight.data.copy_(w1)
            w2 = src.fc2.weight.data[keep_out][:, mid_keep]
            if w2.shape == dst.fc2.weight.shape:
                dst.fc2.weight.data.copy_(w2)
        except Exception as e:
            print(f"[warn] CA slice fallback ({e}); student CA keeps init")


def transfer_res_block(src: nn.Module, dst: nn.Module, keep_in: Optional[torch.Tensor], keep_out: torch.Tensor):
    _slice_conv(src.conv1, dst.conv1, keep_out, keep_in)
    _slice_bn(src.bn1, dst.bn1, keep_out)
    _slice_conv(src.conv2, dst.conv2, keep_out, keep_out)
    _slice_bn(src.bn2, dst.bn2, keep_out)
    _slice_ca(src.ca, dst.ca, keep_out)
    # spatial attention: copy as-is (independent of C)
    with torch.no_grad():
        dst.sa.conv1.weight.data.copy_(src.sa.conv1.weight.data)
    if src.shortcut is not None and dst.shortcut is not None:
        _slice_conv(src.shortcut[0], dst.shortcut[0], keep_out, keep_in)
        _slice_bn(src.shortcut[1], dst.shortcut[1], keep_out)


def _expand_keep_for_layout(layout: List[Optional[int]], keep_idx: Dict[int, torch.Tensor], in_rgb: int = 3):
    """Build keep_in indices for a concat layout of stage ids."""
    pieces = []
    offset = 0
    # For RGB input, keep all 3
    if layout == [None]:
        return torch.arange(in_rgb)

    # teacher concat channel ranges by stage full widths — need teacher nb_filter
    return None  # filled by caller with teacher widths


def build_keep_in(
    layout: List[Optional[int]],
    teacher_nb: Sequence[int],
    keep_idx: Dict[int, torch.Tensor],
) -> torch.Tensor:
    pieces = []
    cursor = 0
    for sid in layout:
        if sid is None:
            pieces.append(torch.arange(3) + cursor)
            cursor += 3
            continue
        width = teacher_nb[sid]
        local = keep_idx[sid]
        pieces.append(local + cursor)
        cursor += width
    return torch.cat(pieces, dim=0)


def transfer_sequential_blocks(
    src_seq: nn.Sequential,
    dst_seq: nn.Sequential,
    keep_in: Optional[torch.Tensor],
    keep_out: torch.Tensor,
):
    assert len(src_seq) == len(dst_seq)
    for i, (sb, db) in enumerate(zip(src_seq, dst_seq)):
        if i == 0:
            transfer_res_block(sb, db, keep_in, keep_out)
        else:
            # same width blocks: in==out==keep_out
            transfer_res_block(sb, db, keep_out, keep_out)


def build_pruned_dnanet(
    teacher: DNANet,
    keep_idx: Dict[int, torch.Tensor],
    num_blocks: Sequence[int],
    input_channels: int = 3,
    num_classes: int = 1,
    deep_supervision=True,
    block=Res_CBAM_block,
) -> DNANet:
    teacher_nb = []
    # infer teacher widths from encoder
    teacher_nb = [
        teacher.conv0_0[0].conv2.out_channels,
        teacher.conv1_0[0].conv2.out_channels,
        teacher.conv2_0[0].conv2.out_channels,
        teacher.conv3_0[0].conv2.out_channels,
        teacher.conv4_0[0].conv2.out_channels,
    ]
    student_nb = [int(keep_idx[s].numel()) for s in range(5)]
    print(f"[prune] teacher nb_filter={teacher_nb}")
    print(f"[prune] student nb_filter={student_nb}")

    student = DNANet(
        num_classes=num_classes,
        input_channels=input_channels,
        block=block,
        num_blocks=list(num_blocks),
        nb_filter=student_nb,
        deep_supervision=deep_supervision,
    )

    # transfer each named layer
    layer_names = list(DNA_NET_IN_LAYOUT.keys())
    for name in layer_names:
        if not hasattr(teacher, name) or not hasattr(student, name):
            continue
        src = getattr(teacher, name)
        dst = getattr(student, name)
        layout = DNA_NET_IN_LAYOUT[name]

        if name.endswith("_1x1"):
            # Conv2d(Cs -> C0): slice in by stage s, out by stage 0
            keep_in = keep_idx[layout[0]]
            keep_out = keep_idx[0]
            _slice_conv(src, dst, keep_out, keep_in)
            continue
        if name.startswith("final"):
            # Conv2d(C0 -> num_classes): slice in only
            keep_in = keep_idx[0]
            _slice_conv(src, dst, None, keep_in)
            continue

        # Sequential of blocks
        keep_in = build_keep_in(layout, teacher_nb, keep_idx)
        # output stage from first block out channels of teacher
        out_c = src[0].conv2.out_channels
        out_stage = teacher_nb.index(out_c)
        keep_out = keep_idx[out_stage]
        transfer_sequential_blocks(src, dst, keep_in, keep_out)

    return student


def count_parameters(model: nn.Module) -> float:
    return sum(p.numel() for p in model.parameters()) / 1e6
