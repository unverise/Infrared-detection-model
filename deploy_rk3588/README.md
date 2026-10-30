# DNANet → RK3588 部署与板端测速

本目录独立于训练代码，专门做：**ONNX 导出 → RKNN 转换 → RK3588 推理延迟 / 内存占用粗测**。

剪枝蒸馏后的学生模型在 NUAA / IRSTD-1k / SIRST-V2 上 IoU 相对教师基本持平或略优；NUDT 需用 `--train_aug resize`（必要时略降剪枝率）。本文件夹用于验证 **板端实际时延与资源占用**，不替代精度表。

---

## 目录结构

```
deploy_rk3588/
├── README.md                 # 本说明
├── export_onnx.py            # PC：权重 → ONNX（单输出）
├── convert_rknn.py           # PC：ONNX → RKNN（需 rknn-toolkit2）
├── bench_onnx_pc.py          # PC：ONNXRuntime 对照延迟
├── configs/
│   ├── model_paths.example.yaml
│   └── calib_list.txt.example
├── board/
│   ├── rknn_infer_bench.py   # 板端：RKNN 延迟 + mem/NPU 摘要
│   └── collect_system_stats.sh
└── outputs/                  # 放 .onnx / .rknn / csv（gitignore 大文件可自行加）
```

---

## 流程概览

```
.pth.tar  --export_onnx.py-->  .onnx  --convert_rknn.py-->  .rknn
                                                      |
                                      拷到 RK3588 板端 |
                                                      v
                              board/rknn_infer_bench.py  (+ collect_system_stats.sh)
```

---

## 1. PC：导出 ONNX

在项目根或本目录执行（需能 import `Infrared-Small-Target-Detection-master`）：

```bash
cd deploy_rk3588

# 学生（推荐：ckpt 内应含 nb_filter）
python export_onnx.py \
  --ckpt ../Infrared-Small-Target-Detection-master/compress_runs/<你的run>/student_best.pth.tar \
  --out outputs/student_nuaa.onnx \
  --input_size 256

# 教师对照
python export_onnx.py \
  --ckpt ../Infrared-Small-Target-Detection-master/result/<教师目录>/mIoU____DNANet_*.pth.tar \
  --out outputs/teacher_nuaa.onnx
```

学生若 ckpt 无 `nb_filter`，加：`--nb_filter 13,22,38,64,102`（按实际剪枝结果改）。

可选 PC 对照：

```bash
pip install onnxruntime
python bench_onnx_pc.py --onnx outputs/student_nuaa.onnx --iters 100
```

---

## 2. PC：转 RKNN

需安装 [rknn-toolkit2](https://github.com/airockchip/rknn-toolkit2)（建议独立 conda，Python 版本按官方矩阵）。

```bash
# FP16 / 无量化（先跑通）
python convert_rknn.py \
  --onnx outputs/student_nuaa.onnx \
  --out outputs/student_nuaa.rknn

# INT8（需校准列表）
cp configs/calib_list.txt.example configs/calib_list.txt
# 编辑填入图片路径后:
python convert_rknn.py \
  --onnx outputs/student_nuaa.onnx \
  --out outputs/student_nuaa_int8.rknn \
  --quantize --dataset configs/calib_list.txt
```

将 `.rknn` 用 `scp` 拷到板子，例如：

```bash
scp outputs/student_nuaa.rknn root@<板IP>:/home/root/dnanet/
scp board/rknn_infer_bench.py board/collect_system_stats.sh root@<板IP>:/home/root/dnanet/
```

---

## 3. 板端：推理速度 + 内存/NPU

板端安装：

```bash
pip3 install rknn-toolkit-lite2 numpy pillow
```

测延迟（默认 NPU 三核 `core_mask=7`）：

```bash
cd /home/root/dnanet
python3 rknn_infer_bench.py \
  --rknn student_nuaa.rknn \
  --warmup 30 --iters 200 \
  --csv bench_student.csv \
  --sample_sys
```

并行采系统资源（内存 Available、NPU load 等）：

```bash
bash collect_system_stats.sh 0.5 sys_stats.csv &
python3 rknn_infer_bench.py --rknn student_nuaa.rknn --iters 200
kill %1
```

输出关注：

| 字段 | 含义 |
|------|------|
| `latency_mean_ms` / `p50` / `p99` | 单帧推理时延 |
| `fps` | 约等于 1000/mean |
| `rknn_mb` | 模型文件体积 |
| `MemAvailable_kB` | 跑完后可用内存快照 |
| `npu_load_raw` | 若固件开放 debug 节点则有 NPU 负载 |

**关于内存带宽**：板端脚本只能给 **MemAvailable / Cached / NPU load** 等间接指标；精确 DDR 带宽需厂商工具（如 Rockchip 性能工具）或示波器。写论文时可写「板端推理时延 + 进程期内存占用 + NPU 负载」，带宽作定性补充。

教师 / 学生各测一遍，同一 `input_size`、同一 `core_mask`，结果写进 `outputs/` 或板端 csv。

---

## 4. 建议记录表（论文/报告）

| 模型 | 平台 | 精度 | 延迟 ms | FPS | 模型体积 MB | MemAvailable | 备注 |
|------|------|------|---------|-----|-------------|--------------|------|
| DNANet teacher | RK3588 NPU | FP16/INT8 | | | | | core_mask=7 |
| DNANet student | RK3588 NPU | FP16/INT8 | | | | | 剪枝率 x |

---

## 依赖速查

| 步骤 | 环境 | 包 |
|------|------|-----|
| export_onnx | 训练同环境 PC | torch |
| convert_rknn | PC（toolkit2） | rknn-toolkit2 |
| bench 板端 | RK3588 | rknn-toolkit-lite2, numpy |

版本请与你板卡系统镜像匹配的 **RKNN Toolkit2 / Lite2** 发布说明对齐。
