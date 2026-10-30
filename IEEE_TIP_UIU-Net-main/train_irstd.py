"""
UIU-Net 全量训练（适配 RDANet-main/datasets 布局: images/ masks/ trainval.txt test.txt）

用法示例:
  python train_irstd.py --dataset NUAA-SIRST --dataset_dir ../RDANet-main/datasets
  python train_irstd.py --dataset NUDT-SIRST --epochs 500 --batch_size 3
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from data_loader import RescaleT, RandomCrop
from model import UIUNET


class ToTensorFloat(object):
    """替代 ToTensorLab：始终输出 float32，避免空 mask 仍为 uint8 导致 collate 失败。"""

    def __call__(self, sample):
        imidx, image, label = sample["imidx"], sample["image"], sample["label"]

        image = image.astype(np.float64)
        label = label.astype(np.float64)

        max_img = float(np.max(image)) if image.size else 0.0
        if max_img > 1e-6:
            image = image / max_img
        max_lbl = float(np.max(label)) if label.size else 0.0
        if max_lbl > 1e-6:
            label = label / max_lbl

        tmp_img = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.float32)
        if image.shape[2] == 1:
            ch = (image[:, :, 0] - 0.485) / 0.229
            tmp_img[:, :, 0] = ch
            tmp_img[:, :, 1] = ch
            tmp_img[:, :, 2] = ch
        else:
            tmp_img[:, :, 0] = (image[:, :, 0] - 0.485) / 0.229
            tmp_img[:, :, 1] = (image[:, :, 1] - 0.456) / 0.224
            tmp_img[:, :, 2] = (image[:, :, 2] - 0.406) / 0.225

        tmp_img = np.ascontiguousarray(tmp_img.transpose(2, 0, 1))
        tmp_lbl = np.ascontiguousarray(label.transpose(2, 0, 1).astype(np.float32))

        return {
            "imidx": torch.from_numpy(np.asarray(imidx)),
            "image": torch.from_numpy(tmp_img),
            "label": torch.from_numpy(tmp_lbl),
        }


def parse_args():
    p = argparse.ArgumentParser(description="Train UIU-Net on IRSTD datasets")
    p.add_argument("--dataset", type=str, required=True,
                   choices=["NUAA-SIRST", "NUDT-SIRST", "IRSTD-1k", "SIRST-V2"])
    p.add_argument("--dataset_dir", type=str, default="../RDANet-main/datasets")
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--batch_size", type=int, default=3, help="官方默认 3；显存紧可改 2")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--rescale", type=int, default=320, help="官方 RescaleT(320)")
    p.add_argument("--crop", type=int, default=288, help="官方 RandomCrop(288)")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--eval_every", type=int, default=10, help="每 N epoch 在 test 上算 IoU")
    p.add_argument("--save_dir", type=str, default="saved_models/uiunet_irstd")
    p.add_argument("--gpus", type=str, default="0")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def read_list(txt_path):
    names = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            # 允许带后缀
            if s.lower().endswith((".png", ".jpg", ".bmp", ".jpeg")):
                s = os.path.splitext(s)[0]
            names.append(s)
    return names


def resolve_path(root, sub, stem):
    for ext in (".png", ".bmp", ".jpg", ".jpeg", ".PNG", ".BMP"):
        p = os.path.join(root, sub, stem + ext)
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(f"missing {sub}/{stem} under {root}")


class IRSTDSalDataset(Dataset):
    """输出格式兼容 SalObjDataset + ToTensorLab。"""

    def __init__(self, root, names, transform=None):
        self.root = root
        self.names = names
        self.transform = transform
        self.img_paths = [resolve_path(root, "images", n) for n in names]
        self.lbl_paths = [resolve_path(root, "masks", n) for n in names]

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        from skimage import io

        image = io.imread(self.img_paths[idx])
        label = io.imread(self.lbl_paths[idx])
        if len(image.shape) == 2:
            image = image[:, :, None]
        if len(label.shape) == 3:
            label = label[:, :, 0]
        label = label[:, :, None]

        sample = {
            "imidx": np.array([idx]),
            "image": image,
            "label": label,
        }
        if self.transform is not None:
            sample = self.transform(sample)
        return sample


def soft_iou(pred, target, thr=0.5):
    """pred/target: (B,1,H,W) 概率或 0/1。"""
    p = (pred > thr).float()
    t = (target > 0.5).float()
    inter = (p * t).sum(dim=(1, 2, 3))
    union = p.sum(dim=(1, 2, 3)) + t.sum(dim=(1, 2, 3)) - inter
    iou = (inter + 1e-6) / (union + 1e-6)
    return float(iou.mean().item())


def build_loaders(args, root):
    train_names = read_list(os.path.join(root, "trainval.txt"))
    test_names = read_list(os.path.join(root, "test.txt"))

    train_tf = transforms.Compose([
        RescaleT(args.rescale),
        RandomCrop(args.crop),
        ToTensorFloat(),
    ])
    test_tf = transforms.Compose([
        RescaleT(args.rescale),
        ToTensorFloat(),
    ])

    train_set = IRSTDSalDataset(root, train_names, transform=train_tf)
    test_set = IRSTDSalDataset(root, test_names, transform=test_tf)

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        drop_last=True,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=1,
        shuffle=False,
        num_workers=max(1, args.workers // 2),
        drop_last=False,
    )
    print(f"[data] train={len(train_set)} test={len(test_set)} "
          f"batch={args.batch_size} rescale={args.rescale} crop={args.crop}")
    return train_loader, test_loader


def muti_bce_loss(bce, d0, d1, d2, d3, d4, d5, d6, labels):
    losses = [bce(d, labels) for d in (d0, d1, d2, d3, d4, d5, d6)]
    total = sum(losses)
    return losses[0], total


@torch.no_grad()
def evaluate(net, loader, device):
    net.eval()
    ious = []
    for data in loader:
        inputs = data["image"].type(torch.FloatTensor).to(device)
        labels = data["label"].type(torch.FloatTensor).to(device)
        d0, *_ = net(inputs)
        # d0 已是 sigmoid 概率（UIU 输出）
        ious.append(soft_iou(d0, labels))
    net.train()
    return float(np.mean(ious)) if ious else 0.0


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    root = os.path.join(args.dataset_dir, args.dataset)
    if not os.path.isdir(root):
        raise SystemExit(f"dataset root not found: {root}")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    work_dir = os.path.join(args.save_dir, f"{args.dataset}_{stamp}")
    os.makedirs(work_dir, exist_ok=True)
    with open(os.path.join(work_dir, "args.txt"), "w", encoding="utf-8") as f:
        for k, v in vars(args).items():
            f.write(f"{k}={v}\n")
    print(f"[work_dir] {work_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = build_loaders(args, root)

    net = UIUNET(3, 1).to(device)
    bce = nn.BCELoss(reduction="mean")
    optimizer = optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    best_iou = -1.0
    log_path = os.path.join(work_dir, "train_log.csv")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("epoch,train_loss,tar_loss,test_iou,lr\n")

    for epoch in range(1, args.epochs + 1):
        net.train()
        run_loss = 0.0
        run_tar = 0.0
        n_batches = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", ncols=100)
        for data in pbar:
            inputs = data["image"].type(torch.FloatTensor).to(device)
            labels = data["label"].type(torch.FloatTensor).to(device)

            optimizer.zero_grad()
            d0, d1, d2, d3, d4, d5, d6 = net(inputs)
            loss_tar, loss = muti_bce_loss(bce, d0, d1, d2, d3, d4, d5, d6, labels)
            loss.backward()
            optimizer.step()

            run_loss += float(loss.item())
            run_tar += float(loss_tar.item())
            n_batches += 1
            pbar.set_postfix(loss=f"{run_loss/n_batches:.4f}", tar=f"{run_tar/n_batches:.4f}")

        scheduler.step()
        avg_loss = run_loss / max(1, n_batches)
        avg_tar = run_tar / max(1, n_batches)
        lr = optimizer.param_groups[0]["lr"]

        test_iou = -1.0
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            test_iou = evaluate(net, test_loader, device)
            print(f"[eval] epoch={epoch} test_IoU={test_iou:.4f} best={max(best_iou, test_iou):.4f}")
            if test_iou > best_iou:
                best_iou = test_iou
                ckpt = {
                    "epoch": epoch,
                    "state_dict": net.state_dict(),
                    "best_iou": best_iou,
                    "args": vars(args),
                }
                path = os.path.join(work_dir, "uiunet_best.pth.tar")
                torch.save(ckpt, path)
                print(f"[save] best IoU={best_iou:.4f} -> {path}")

        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"{epoch},{avg_loss:.6f},{avg_tar:.6f},{test_iou:.6f},{lr:.8f}\n")

        # 定期备份
        if epoch % 50 == 0:
            torch.save(
                {"epoch": epoch, "state_dict": net.state_dict(), "best_iou": best_iou},
                os.path.join(work_dir, f"uiunet_epoch_{epoch}.pth.tar"),
            )

    print(f"[done] dataset={args.dataset} best_IoU={best_iou:.4f} dir={work_dir}")


if __name__ == "__main__":
    main()
