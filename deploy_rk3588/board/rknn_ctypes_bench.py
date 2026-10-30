#!/usr/bin/env python3
"""
RK3588 板端测速（仅标准库 + /usr/lib/librknnrt.so，无需 pip / rknnlite / numpy）。

用法（板上）:
  python3 rknn_ctypes_bench.py --rknn student.rknn --warmup 30 --iters 200
  python3 rknn_ctypes_bench.py --rknn student.rknn --csv bench.csv
"""

from __future__ import annotations

import argparse
import csv
import ctypes
import os
import struct
import sys
import time
from datetime import datetime

LIB_PATH = "/usr/lib/librknnrt.so"

RKNN_TENSOR_UINT8 = 3
RKNN_TENSOR_FLOAT32 = 0
RKNN_TENSOR_FLOAT16 = 1
RKNN_TENSOR_NHWC = 1
RKNN_TENSOR_NCHW = 0

RKNN_QUERY_IN_OUT_NUM = 0
RKNN_QUERY_INPUT_ATTR = 1
RKNN_QUERY_OUTPUT_ATTR = 2
RKNN_QUERY_SDK_VERSION = 5

# core mask（部分驱动在 rknn_init flag 里体现；此处先用默认 0，兼容性最好）
RKNN_FLAG_PRIOR_HIGH = 0


class RKNNIONum(ctypes.Structure):
    _fields_ = [("n_input", ctypes.c_uint32), ("n_output", ctypes.c_uint32)]


class RKNNTensorAttr(ctypes.Structure):
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
    _fields_ = [
        ("index", ctypes.c_uint32),
        ("buf", ctypes.c_void_p),
        ("size", ctypes.c_uint32),
        ("pass_through", ctypes.c_uint8),
        ("type", ctypes.c_uint8),
        ("fmt", ctypes.c_uint8),
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


def make_input_buffer(in_attr, fill="random"):
    """按模型输入 attr 构造 buffer；默认随机，测速足够。"""
    n = int(in_attr.n_elems)
    size = int(in_attr.size)
    t = int(in_attr.type)
    # type: 0=fp32, 1=fp16, 2=int8, 3=uint8, ...
    if t == RKNN_TENSOR_UINT8 or size == n:
        if fill == "ones":
            raw = bytes([1]) * n
        else:
            # 简单伪随机，避免依赖 random 模块大开销也可；用 deterministic pattern
            raw = bytes((i * 37 + 11) & 0xFF for i in range(n))
        return raw, RKNN_TENSOR_UINT8, 0  # pass_through=0 让 runtime 做必要转换
    if t == RKNN_TENSOR_FLOAT16 or size == n * 2:
        vals = [1.0] * n
        raw = struct.pack("<%de" % n, *vals)
        return raw, RKNN_TENSOR_FLOAT16, 1
    # fallback float32
    vals = [1.0] * n
    raw = struct.pack("<%df" % n, *vals)
    return raw, RKNN_TENSOR_FLOAT32, 0


def main():
    p = argparse.ArgumentParser(description="RK3588 RKNN ctypes latency bench")
    p.add_argument("--rknn", required=True)
    p.add_argument("--warmup", type=int, default=30)
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--csv", default="bench_result.csv")
    p.add_argument("--lib", default=LIB_PATH)
    args = p.parse_args()

    if not os.path.isfile(args.rknn):
        raise SystemExit("missing rknn: %s" % args.rknn)
    if not os.path.isfile(args.lib):
        raise SystemExit("missing runtime: %s" % args.lib)

    lib = ctypes.CDLL(args.lib)
    bind_lib(lib)

    ctx = ctypes.c_uint64(0)
    path = ctypes.c_char_p(os.path.abspath(args.rknn).encode("utf-8"))
    check(lib.rknn_init(ctypes.byref(ctx), path, 0, RKNN_FLAG_PRIOR_HIGH, None), "rknn_init")

    version = RKNNSdkVersion()
    check(
        lib.rknn_query(ctx.value, RKNN_QUERY_SDK_VERSION, ctypes.byref(version), ctypes.sizeof(version)),
        "query sdk",
    )
    print("api:", c_string(version.api_version))
    print("drv:", c_string(version.drv_version))

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
    )

    raw, dtype, pass_through = make_input_buffer(in_attr)
    buf = ctypes.create_string_buffer(raw)
    inp = RKNNInput()
    inp.index = 0
    inp.buf = ctypes.cast(buf, ctypes.c_void_p)
    inp.size = len(raw)
    inp.pass_through = pass_through
    inp.type = dtype
    inp.fmt = in_attr.fmt if in_attr.fmt in (0, 1) else RKNN_TENSOR_NHWC

    def one_run():
        check(lib.rknn_inputs_set(ctx.value, 1, ctypes.byref(inp)), "inputs_set")
        check(lib.rknn_run(ctx.value, None), "run")
        out = RKNNOutput()
        out.want_float = 1
        out.is_prealloc = 0
        out.index = 0
        check(lib.rknn_outputs_get(ctx.value, 1, ctypes.byref(out), None), "outputs_get")
        lib.rknn_outputs_release(ctx.value, 1, ctypes.byref(out))

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

    # latency 明细
    detail = os.path.splitext(args.csv)[0] + "_latencies.txt"
    with open(detail, "w") as f:
        for x in lat:
            f.write("%.4f\n" % x)
    print("[save]", os.path.abspath(detail))

    lib.rknn_destroy(ctx.value)


if __name__ == "__main__":
    main()
