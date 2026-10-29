"""IRSTD transferable compression package (DNANet first adapter)."""

__all__ = [
    "PrunableGroup",
    "ImportanceEstimator",
    "build_pruned_dnanet",
    "save_keep_masks",
    "RegionSoftIoULoss",
    "kd_logits_loss",
]


def __getattr__(name):
    if name == "PrunableGroup":
        from .groups import PrunableGroup
        return PrunableGroup
    if name == "ImportanceEstimator":
        from .importance import ImportanceEstimator
        return ImportanceEstimator
    if name in ("build_pruned_dnanet", "save_keep_masks"):
        from . import prune as _prune
        return getattr(_prune, name)
    if name in ("RegionSoftIoULoss", "kd_logits_loss"):
        from . import losses as _losses
        return getattr(_losses, name)
    raise AttributeError(name)
