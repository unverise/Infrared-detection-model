# DNANet 可迁移压缩训练说明

本目录新增 `compress/` 包与入口 `compress_train.py`，对应方案：

1. **P1 敏感性评分**（区域 SoftIoU 权重梯度 + 局部对比）
2. **P2 结构化剪枝**（按 stage 共掩码缩小 `nb_filter`，重建学生并拷贝权重）
3. **P3 蒸馏重训**（SoftIoU + 区域 SoftIoU + 输出 KD）

**未改 DNANet 嵌套拓扑**；只压缩各尺度通道宽度。

---

## 0. 先训教师（原流程，不变）

```bash
cd Infrared-Small-Target-Detection-master
python train.py --dataset IRSTD-1k --root ../RDANet-main/datasets --epochs 1500 --deep_supervision True --train_batch_size 4
```

教师权重一般在 `result/.../mIoU__DNANet_*_epoch.pth.tar`。

---

## 1. 一键压缩流水线（评分 → 剪枝 → 蒸馏）

```bash
python compress_train.py --mode pipeline ^
  --teacher_ckpt result/你的教师权重.pth.tar ^
  --dataset IRSTD-1k ^
  --root ../RDANet-main/datasets ^
  --epochs 200 ^
  --train_batch_size 4 ^
  --score_batches 50 ^
  --prune_rates 0.20,0.30,0.40,0.50,0.60 ^
  --work_dir ./compress_runs
```

Linux/macOS 把 `^` 换成 `\`。

输出目录示例：`compress_runs/IRSTD-1k_pipeline_时间戳/`

| 文件 | 含义 |
|------|------|
| `scores.pt` | 各 stage 通道重要性 |
| `keep_masks.pt` | 保留通道下标 |
| `student_init.pth.tar` | 剪枝后、蒸馏前学生 |
| `student_best.pth.tar` | 蒸馏过程最佳学生 |
| `distill_log.txt` | loss / mIoU 日志 |
| `args.json` | 本次超参 |

---

## 2. 分步跑（可选）

```bash
# 只评分
python compress_train.py --mode score --teacher_ckpt TEACHER.pth.tar --dataset IRSTD-1k --root ../RDANet-main/datasets

# 只剪枝（需 keep_masks）
python compress_train.py --mode prune --teacher_ckpt TEACHER.pth.tar --keep_path xxx/keep_masks.pt ...

# 只蒸馏
python compress_train.py --mode distill --teacher_ckpt TEACHER.pth.tar --keep_path xxx/keep_masks.pt --student_ckpt xxx/student_init.pth.tar --epochs 200 ...
```

---

## 3. 关键超参

| 参数 | 默认 | 说明 |
|------|------|------|
| `--prune_rates` | `0.2,0.3,0.4,0.5,0.6` | stage0→4 剪枝率（浅少深多） |
| `--alpha` | `0.7` | 任务梯度 vs 对比项融合 |
| `--lambda_seg/kd/region` | `1/1/0.5` | 蒸馏损失权重 |
| `--score_batches` | `50` | 估分用多少个 batch |

---

## 4. 建议训练顺序（对应论文实验）

1. 训满教师 DNA-Net，记录 mIoU / Params  
2. `pipeline` 出学生，对比教师 Params 与 mIoU  
3. 改 `--prune_rates` 做消融；或把评分换成随机（后续可加）  
4. 同样协议以后接到 UIU / SCTrans（只需新 Adapter）

---

## 5. 注意

- 编译/运行目录必须是 `Infrared-Small-Target-Detection-master`（才能 `import model` / `import compress`）。  
- `deep_supervision` 请与教师训练时一致（默认 `True`）。  
- 若 `load_state_dict` 缺 key，脚本会 `strict=False` 并打印 warn，请核对 ckpt 是否为 DNANet。  
