# ADB 部署 DNANet 到 RK3588 测速（按这个做）

你这边已经 `adb devices` 能看到板子。完整链路是：

```
本机/训练机: .pth.tar → .onnx → .rknn
                    ↓ adb push
              RK3588: 跑 ctypes 测速脚本 → 得到延迟 / 内存 / NPU 负载
```

---

## 当前环境结论（已检查）

| 位置 | 状态 |
|------|------|
| ADB | 已连接 `ok3588`（`64cce07a13a37422`） |
| 板端 | 有 `/usr/lib/librknnrt.so`，**无 pip / 无 rknnlite** → 用 `board/rknn_ctypes_bench.py` |
| 本机 Windows | Python 3.14，**无 torch、无 rknn-toolkit2** → **不能在本机直接转 RKNN** |
| 权重 | 本地已有各数据集 `student_best.pth.tar` 与教师权重 |

因此：**导出 ONNX + 转 RKNN 请在训练机（AutoDL）或另装 Python3.8~3.10 + toolkit2 的环境做**；本机负责 `adb push` 和拉回结果。

---

## 步骤 0：板端冒烟（确认 NPU 可用，1 分钟）

板子上已有 `tiny.rknn` 和 `board_npu_test.py`：

```powershell
adb shell "cd /userdata && python3 board_npu_test.py"
```

看到 `PASS` 再继续。

---

## 步骤 1：在训练机导出 ONNX（有 torch 的环境）

以 NUAA 学生为例（路径按你实际 run 改）：

```bash
cd deploy_rk3588

python export_onnx.py \
  --ckpt ../Infrared-Small-Target-Detection-master/compress_runs/NUAA-SIRST_pipeline_20261029_181406/student_best.pth.tar \
  --out outputs/student_nuaa.onnx \
  --input_size 256
```

教师对照：

```bash
python export_onnx.py \
  --ckpt ../Infrared-Small-Target-Detection-master/result/NUAA-SIRST_DNANet_14_10_2026_11_07_43_wDS/mIoU__DNANet_NUAA-SIRST_epoch.pth.tar \
  --out outputs/teacher_nuaa.onnx \
  --input_size 256
```

把 `outputs/*.onnx` 下载到本机 `deploy_rk3588/outputs/`。

---

## 步骤 2：转 RKNN（必须有 rknn-toolkit2）

推荐独立 conda：`Python 3.8/3.9/3.10`（与 [rknn-toolkit2](https://github.com/airockchip/rknn-toolkit2) 版本矩阵一致），**不要用 3.14**。

```bash
cd deploy_rk3588
# 先跑通：不做 INT8
python convert_rknn.py \
  --onnx outputs/student_nuaa.onnx \
  --out outputs/student_nuaa.rknn
```

得到 `student_nuaa.rknn` 后放到本机 `deploy_rk3588/outputs/`。

> 若暂时没有 toolkit2：在 x86_64 Linux（或官方支持的 Windows 包）装好后再转；**不能在 RK3588 板上把 ONNX 转成 RKNN**。

---

## 步骤 3：ADB 推到板子

在本机 PowerShell：

```powershell
cd "d:\Desktop\红外检测模型\deploy_rk3588"

adb shell "mkdir -p /userdata/dnanet_bench"
adb push outputs\student_nuaa.rknn /userdata/dnanet_bench/
adb push board\rknn_ctypes_bench.py /userdata/dnanet_bench/
adb push board\collect_system_stats.sh /userdata/dnanet_bench/
```

教师模型同样 push 一个 `teacher_nuaa.rknn` 方便对比。

---

## 步骤 4：板端测推理速度 + 内存/NPU

```powershell
# 只测延迟与内存快照
adb shell "cd /userdata/dnanet_bench && python3 rknn_ctypes_bench.py --rknn student_nuaa.rknn --warmup 30 --iters 200 --csv bench_student.csv"
```

并行采系统资源（可选）：

```powershell
adb shell "cd /userdata/dnanet_bench && nohup bash collect_system_stats.sh 0.5 sys_stats.csv >/dev/null 2>&1 & echo \$! > stats.pid"
adb shell "cd /userdata/dnanet_bench && python3 rknn_ctypes_bench.py --rknn student_nuaa.rknn --warmup 30 --iters 200 --csv bench_student.csv"
adb shell "cd /userdata/dnanet_bench && kill \$(cat stats.pid)"
```

拉回结果：

```powershell
adb pull /userdata/dnanet_bench/bench_student.csv .\outputs\
adb pull /userdata/dnanet_bench/bench_student_latencies.txt .\outputs\
adb pull /userdata/dnanet_bench/sys_stats.csv .\outputs\
```

关注字段：`latency_mean_ms` / `p50` / `p99` / `fps` / `rknn_mb` / `MemAvailable_*` / `npu_load_*`。

---

## 步骤 5：教师 vs 学生对比（建议）

同一 `input_size=256`、同样 warmup/iters，各跑一遍，填表：

| 模型 | 延迟 mean ms | FPS | 体积 MB | MemAvailable | NPU load |
|------|--------------|-----|---------|--------------|----------|
| teacher | | | | | |
| student | | | | | |

---

## 关于「内存带宽」

板端脚本能稳定给出：**推理延迟、模型体积、MemAvailable/Cached、NPU load（若 debug 节点可读）**。  
精确 DDR 带宽占用需要 Rockchip 厂商性能工具，一般论文写「板端时延 + 内存占用 + NPU 负载」即可。

---

## 你现在立刻可做的两件事

1. 跑步骤 0 确认 NPU `PASS`
2. 在 AutoDL（有 torch）上跑步骤 1 出 ONNX；有 toolkit2 再步骤 2；然后本机从步骤 3 开始 ADB
