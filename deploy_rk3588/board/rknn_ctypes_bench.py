#!/usr/bin/env python3
"""
RK3588 板端测速（仅标准库 + /usr/lib/librknnrt.so，无需 pip / rknnlite / numpy）。

对齐 rknn-toolkit2 / librknnrt 2.x 的 rknn_api.h 结构体布局。

用法（板上）:
  python3 rknn_ctypes_bench.py --rknn student.rknn --warmup 30 --iters 200
  python3 rknn_ctypes_bench.py --rknn student.rknn --csv bench.csv
"""

from __future__ import annotations

import argparse
import csv
import ctypes
import os
import time
from datetime import datetime

LIB_PATH = "/usr/lib/librknnrt.so"

RKNN_TENSOR_FLOAT32 = 0
RKNN_TENSOR_FLOAT16 = 1
RKNN_TENSOR_UINT8 = 3
RKNN_TENSOR_NHWC = 1
RKNN_TENSOR_NCHW = 0

RKNN_QUERY_IN_OUT_NUM = 0
RKNN_QUERY_INPUT_ATTR = 1
RKNN_QUERY_OUTPUT_ATTR = 2
RKNN_QUERY_SDK_VERSION = 5

RKNN_FLAG_PRIOR_HIGH = 0

# rknn_core_mask（RK3588 多核）
RKNN_NPU_CORE_AUTO = 0
RKNN_NPU_CORE_0 = 1
RKNN_NPU_CORE_1 = 2
RKNN_NPU_CORE_2 = 4
RKNN_NPU_CORE_0_1 = 3
RKNN_NPU_CORE_0_1_2 = 7
RKNN_NPU_CORE_ALL = 0xffff


class RKNNIONum(ctypes.Structure):
    _fields_ = [("n_input", ctypes.c_uint32), ("n_output", ctypes.c_uint32)]


class RKNNTensorAttr(ctypes.Structure):
    # 与 rknn_api.h 中 rknn_tensor_attr 一致；ctypes 默认对齐会在 int8 后补齐
    _fields_ = [
        ("index", ctypes.c_uint32),
        ("n_dims", ctypes.c_uint32),
        ("dims", ctypes.c_uint32 * 16),
        ("name", ctypes.c_char * 256),
        ("n_elems", ctypes.c_uint32),
        ("size", ctypes.c_uint32),
        ("fmt", ctypes.c_int),
        ("type", ctypes.c_int),
        ("qnt_type", ctypes.c_int),
        ("fl", ctypes.c_int8),
        ("zp", ctypes.c_int32),
        ("scale", ctypes.c_float),
        ("w_stride", ctypes.c_uint32),
        ("size_with_stride", ctypes.c_uint32),
        ("pass_through", ctypes.c_uint8),
        ("h_stride", ctypes.c_uint32),
    ]


class RKNNSdkVersion(ctypes.Structure):
    _fields_ = [
        ("api_version", ctypes.c_char * 256),
        ("drv_version", ctypes.c_char * 256),
    ]


class RKNNInput(ctypes.Structure):
    """注意: type/fmt 是 enum(=int)，不是 uint8。旧脚本写成 uint8 会导致 inputs_set 错乱。"""

    _fields_ = [
        ("index", ctypes.c_uint32),
        ("buf", ctypes.c_void_p),
        ("size", ctypes.c_uint32),
        ("pass_through", ctypes.c_uint8),
        ("type", ctypes.c_int),
        ("fmt", ctypes.c_int),
    ]


class RKNNOutput(ctypes.Structure):
    _fields_ = [
        ("want_float", ctypes.c_uint8),
        ("is_prealloc", ctypes.c_uint8),
        ("index", ctypes.c_uint32),
        ("buf", ctypes.c_void_p),
        ("size", ctypes.c_uint32),
    ]


def check(ret, what):
    if ret != 0:
        raise SystemExit("%s failed, ret=%s" % (what, ret))


def c_string(buf):
    return buf.split(b"\x00", 1)[0].decode("utf-8", "replace")


def read_meminfo():
    out = {}
    try:
        with open("/proc/meminfo", "r") as f:
            for line in f:
                parts = line.split(":")
                if len(parts) >= 2:
                    out[parts[0].strip()] = int(parts[1].strip().split()[0])
    except OSError:
        pass
    return out


def read_npu_load():
    path = "/sys/kernel/debug/rknpu/load"
    if not os.path.exists(path):
        return ""
    try:
        with open(path, "r") as f:
            return f.read().strip().replace("\n", " | ")
    except OSError:
        return ""


def percentile(sorted_vals, p):
    if not sorted_vals:
        return 0.0
    k = (len(sorted_vals) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    if f == c:
        return float(sorted_vals[f])
    return float(sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f))


def bind_lib(lib):
    lib.rknn_init.argtypes = [
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_void_p,
    ]
    lib.rknn_init.restype = ctypes.c_int
    lib.rknn_destroy.argtypes = [ctypes.c_uint64]
    lib.rknn_destroy.restype = ctypes.c_int
    lib.rknn_query.argtypes = [
        ctypes.c_uint64,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_uint32,
    ]
    lib.rknn_query.restype = ctypes.c_int
    lib.rknn_inputs_set.argtypes = [
        ctypes.c_uint64,
        ctypes.c_uint32,
        ctypes.POINTER(RKNNInput),
    ]
    lib.rknn_inputs_set.restype = ctypes.c_int
    lib.rknn_run.argtypes = [ctypes.c_uint64, ctypes.c_void_p]
    lib.rknn_run.restype = ctypes.c_int
    lib.rknn_outputs_get.argtypes = [
        ctypes.c_uint64,
        ctypes.c_uint32,
        ctypes.POINTER(RKNNOutput),
        ctypes.c_void_p,
    ]
    lib.rknn_outputs_get.restype = ctypes.c_int
    lib.rknn_outputs_release.argtypes = [
        ctypes.c_uint64,
        ctypes.c_uint32,
        ctypes.POINTER(RKNNOutput),
    ]
    lib.rknn_outputs_release.restype = ctypes.c_int
    if hasattr(lib, "rknn_set_core_mask"):
        lib.rknn_set_core_mask.argtypes = [ctypes.c_uint64, ctypes.c_int]
        lib.rknn_set_core_mask.restype = ctypes.c_int


def make_uint8_nhwc(dims):
    """
    官方 demo 常用路径: 喂 NHWC uint8，pass_through=0，由 runtime 转成模型内部类型。
    dims 一般为 [N,H,W,C] 或 [N,C,H,W]。
    """
    if len(dims) != 4:
        raise SystemExit("unexpected dims: %s" % dims)
    # 启发式: 若 dims[1] 很小(<=4) 且 dims[-1] 较大 -> NCHW，否则按 NHWC
    if dims[1] <= 4 and dims[-1] > 4:
        n, c, h, w = dims
        n_pix = n * h * w * c
        fmt = RKNN_TENSOR_NHWC  # 仍喂 NHWC，让 runtime 转换
        # 构造 NHWC 布局
        raw = bytes((i * 37 + 11) & 0xFF for i in range(n_pix))
    else:
        n, h, w, c = dims
        n_pix = n * h * w * c
        fmt = RKNN_TENSOR_NHWC
        raw = bytes((i * 37 + 11) & 0xFF for i in range(n_pix))
    return raw, RKNN_TENSOR_UINT8, 0, fmt


def main():
    p = argparse.ArgumentParser(description="RK3588 RKNN ctypes latency bench")
    p.add_argument("--rknn", required=True)
    p.add_argument("--warmup", type=int, default=30)
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--csv", default="bench_result.csv")
    p.add_argument("--lib", default=LIB_PATH)
    p.add_argument(
        "--core_mask",
        type=int,
        default=RKNN_NPU_CORE_0_1_2,
        help="NPU core mask: 1/2/4 单核, 7=三核, 0=auto",
    )
    args = p.parse_args()

    if not os.path.isfile(args.rknn):
        raise SystemExit("missing rknn: %s" % args.rknn)
    if not os.path.isfile(args.lib):
        raise SystemExit("missing runtime: %s" % args.lib)

    print("sizeof RKNNInput=", ctypes.sizeof(RKNNInput), "RKNNTensorAttr=", ctypes.sizeof(RKNNTensorAttr))

    lib = ctypes.CDLL(args.lib)
    bind_lib(lib)

    ctx = ctypes.c_uint64(0)
    path = ctypes.c_char_p(os.path.abspath(args.rknn).encode("utf-8"))
    check(lib.rknn_init(ctypes.byref(ctx), path, 0, RKNN_FLAG_PRIOR_HIGH, None), "rknn_init")

    if hasattr(lib, "rknn_set_core_mask"):
        ret = lib.rknn_set_core_mask(ctx.value, int(args.core_mask))
        print("core_mask:", args.core_mask, "ret=", ret)
        if ret != 0:
            print("[warn] rknn_set_core_mask failed, continue with default cores")

    version = RKNNSdkVersion()
    check(
        lib.rknn_query(ctx.value, RKNN_QUERY_SDK_VERSION, ctypes.byref(version), ctypes.sizeof(version)),
        "query sdk",
    )
    print("api:", c_string(version.api_version))
    print("drv:", c_string(version.drv_version))
    if c_string(version.drv_version).startswith("0.7"):
        print("[warn] NPU driver 较旧(0.7.x)，runtime 2.5 官方常建议 >=0.8.8；若仍失败需升级板端固件。")

    io_num = RKNNIONum()
    check(
        lib.rknn_query(ctx.value, RKNN_QUERY_IN_OUT_NUM, ctypes.byref(io_num), ctypes.sizeof(io_num)),
        "query io",
    )
    print("inputs:", io_num.n_input, "outputs:", io_num.n_output)

    in_attr = RKNNTensorAttr()
    in_attr.index = 0
    check(
        lib.rknn_query(ctx.value, RKNN_QUERY_INPUT_ATTR, ctypes.byref(in_attr), ctypes.sizeof(in_attr)),
        "query input",
    )
    dims = list(in_attr.dims[: in_attr.n_dims])
    print(
        "input:",
        c_string(in_attr.name),
        "dims=",
        dims,
        "type=",
        in_attr.type,
        "fmt=",
        in_attr.fmt,
        "size=",
        in_attr.size,
        "n_elems=",
        in_attr.n_elems,
        "size_with_stride=",
        in_attr.size_with_stride,
    )

    raw, dtype, pass_through, fmt = make_uint8_nhwc(dims)
    buf = ctypes.create_string_buffer(raw)

    inputs = (RKNNInput * 1)()
    inputs[0].index = 0
    inputs[0].buf = ctypes.cast(buf, ctypes.c_void_p)
    inputs[0].size = len(raw)
    inputs[0].pass_through = pass_through
    inputs[0].type = dtype
    inputs[0].fmt = fmt
    print(
        "feed: size=%d type=%d(fmt uint8) fmt=%d(NHWC) pass_through=%d"
        % (inputs[0].size, inputs[0].type, inputs[0].fmt, inputs[0].pass_through)
    )

    def one_run():
        check(lib.rknn_inputs_set(ctx.value, 1, inputs), "inputs_set")
        check(lib.rknn_run(ctx.value, None), "run")
        outs = (RKNNOutput * 1)()
        outs[0].want_float = 1
        outs[0].is_prealloc = 0
        outs[0].index = 0
        check(lib.rknn_outputs_get(ctx.value, 1, outs, None), "outputs_get")
        lib.rknn_outputs_release(ctx.value, 1, outs)

    print("warmup:", args.warmup)
    for _ in range(args.warmup):
        one_run()

    mem0 = read_meminfo()
    npu0 = read_npu_load()
    lat = []
    for _ in range(args.iters):
        t0 = time.perf_counter()
        one_run()
        lat.append((time.perf_counter() - t0) * 1000.0)

    mem1 = read_meminfo()
    npu1 = read_npu_load()
    lat_sorted = sorted(lat)
    mean = sum(lat) / float(len(lat))
    var = sum((x - mean) ** 2 for x in lat) / float(len(lat))
    std = var ** 0.5

    summary = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "rknn": os.path.abspath(args.rknn),
        "rknn_mb": round(os.path.getsize(args.rknn) / 1024.0 / 1024.0, 3),
        "input_dims": str(dims),
        "core_mask": args.core_mask,
        "warmup": args.warmup,
        "iters": args.iters,
        "latency_mean_ms": round(mean, 3),
        "latency_std_ms": round(std, 3),
        "latency_p50_ms": round(percentile(lat_sorted, 50), 3),
        "latency_p90_ms": round(percentile(lat_sorted, 90), 3),
        "latency_p99_ms": round(percentile(lat_sorted, 99), 3),
        "fps": round(1000.0 / mean, 2) if mean > 0 else 0,
        "MemAvailable_before_kB": mem0.get("MemAvailable"),
        "MemAvailable_after_kB": mem1.get("MemAvailable"),
        "MemTotal_kB": mem1.get("MemTotal"),
        "Cached_after_kB": mem1.get("Cached"),
        "npu_load_before": npu0,
        "npu_load_after": npu1,
        "api": c_string(version.api_version),
        "drv": c_string(version.drv_version),
    }

    print("======== RK3588 Bench ========")
    for k, v in summary.items():
        print("  %s: %s" % (k, v))
    print("================================")
    print("说明: MemAvailable/Cached/NPU load 为间接资源指标；精确 DDR 带宽需厂商工具。")

    write_header = not os.path.isfile(args.csv)
    with open(args.csv, "a") as f:
        w = csv.DictWriter(f, fieldnames=list(summary.keys()))
        if write_header:
            w.writeheader()
        w.writerow(summary)
    print("[save]", os.path.abspath(args.csv))

    detail = os.path.splitext(args.csv)[0] + "_latencies.txt"
    with open(detail, "w") as f:
        for x in lat:
            f.write("%.4f\n" % x)
    print("[save]", os.path.abspath(detail))

    lib.rknn_destroy(ctx.value)


if __name__ == "__main__":
    main()
