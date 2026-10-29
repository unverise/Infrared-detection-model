import torch
import torch.nn as nn
import torch.nn.functional as F


def soft_iou_loss(pred, target, weight=None, smooth=1.0):
    """
    SoftIoU on logits. Optional spatial weight in [0,1], same shape as target.
    """
    pred = torch.sigmoid(pred)
    if weight is None:
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum() - intersection
        return 1.0 - (intersection + smooth) / (union + smooth)

    w = weight
    intersection = (pred * target * w).sum()
    union = (pred * w).sum() + (target * w).sum() - intersection
    return 1.0 - (intersection + smooth) / (union + smooth)


def make_target_region_weight(mask, dilate=3):
    """
    Build a spatial weight map that emphasizes target neighborhood.
    mask: (B,1,H,W) in {0,1} or [0,1]
    """
    if dilate <= 0:
        return torch.clamp(mask, 0.0, 1.0)

    # avg_pool as cheap dilation approximation on soft/hard masks
    k = 2 * dilate + 1
    pooled = F.avg_pool2d(mask, kernel_size=k, stride=1, padding=dilate)
    region = (pooled > 0).float()
    # background gets small weight so gradients are not entirely zero outside
    weight = 0.05 + 0.95 * region
    return weight


class RegionSoftIoULoss(nn.Module):
    def __init__(self, dilate=3):
        super().__init__()
        self.dilate = dilate

    def forward(self, pred, target):
        weight = make_target_region_weight(target, self.dilate)
        return soft_iou_loss(pred, target, weight=weight)


def kd_logits_loss(student_logits, teacher_logits, T=1.0):
    """MSE on sigmoid masks (stable for segmentation KD)."""
    s = torch.sigmoid(student_logits / T)
    t = torch.sigmoid(teacher_logits / T).detach()
    return F.mse_loss(s, t)


def prediction_loss(preds, labels, criterion, deep_supervision=True):
    if deep_supervision and isinstance(preds, (list, tuple)):
        loss = 0.0
        for p in preds:
            loss = loss + criterion(p, labels)
        return loss / len(preds)
    if isinstance(preds, (list, tuple)):
        preds = preds[-1]
    return criterion(preds, labels)


def take_final(preds):
    if isinstance(preds, (list, tuple)):
        return preds[-1]
    return preds
