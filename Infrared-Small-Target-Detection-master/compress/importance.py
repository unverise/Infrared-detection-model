from __future__ import annotations

from typing import Dict, List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from compress.adapters.dnanet_adapter import DNANetAdapter
from compress.losses import RegionSoftIoULoss, take_final


class ImportanceEstimator:
    """
    P1: I = alpha * I_task + (1-alpha) * I_contrast

    I_task  : mean |grad| of Res_CBAM conv2 weights (per out-channel), aggregated by stage
    I_contrast: target-neighborhood activation minus background (per channel)
    Scores are shared per stage (co-mask).
    """

    def __init__(
        self,
        adapter: DNANetAdapter,
        alpha: float = 0.7,
        dilate: int = 3,
        max_batches: int = 50,
    ):
        self.adapter = adapter
        self.alpha = alpha
        self.dilate = dilate
        self.max_batches = max_batches
        self.region_loss = RegionSoftIoULoss(dilate=dilate)

    @torch.no_grad()
    def _contrast_score_from_feat(self, feat: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        b, c, h, w = feat.shape
        m = F.interpolate(mask, size=(h, w), mode="nearest")
        if self.dilate > 0:
            k = 2 * self.dilate + 1
            m = (F.avg_pool2d(m, k, stride=1, padding=self.dilate) > 0).float()
        bg = 1.0 - m
        tgt_den = m.sum(dim=(2, 3)).clamp_min(1.0)
        bg_den = bg.sum(dim=(2, 3)).clamp_min(1.0)
        tgt = (feat * m).sum(dim=(2, 3)) / tgt_den
        bga = (feat * bg).sum(dim=(2, 3)) / bg_den
        return (tgt - bga).mean(dim=0)

    def estimate(self, model: nn.Module, dataloader, device="cuda") -> Dict[str, torch.Tensor]:
        model.train()  # enable grad on weights
        num_stages = self.adapter.num_stages
        widths = self.adapter.nb_filter
        task_scores = [torch.zeros(widths[s], device=device) for s in range(num_stages)]
        contrast_scores = [torch.zeros(widths[s], device=device) for s in range(num_stages)]
        task_counts = [0 for _ in range(num_stages)]
        contrast_counts = [0 for _ in range(num_stages)]

        hook_modules = self.adapter.get_hook_modules(model)
        feat_cache: Dict[int, List[torch.Tensor]] = {s: [] for s in range(num_stages)}
        handles = []

        def make_hook(stage_id):
            def _hook(module, inp, out):
                feat_cache[stage_id].append(out.detach())
            return _hook

        for stage, modules in hook_modules.items():
            for m in modules:
                handles.append(m.register_forward_hook(make_hook(stage)))

        n_batches = 0
        pbar = tqdm(dataloader, desc="Importance scoring")
        for data, labels in pbar:
            if n_batches >= self.max_batches:
                break
            data = data.to(device)
            labels = labels.to(device)
            for s in feat_cache:
                feat_cache[s] = []

            model.zero_grad(set_to_none=True)
            preds = model(data)
            loss = self.region_loss(take_final(preds), labels)
            loss.backward()

            # task score from conv2.weight.grad of each hooked block
            for stage, modules in hook_modules.items():
                stage_vec = torch.zeros(widths[stage], device=device)
                used = 0
                for m in modules:
                    g = m.conv2.weight.grad
                    if g is None:
                        continue
                    # [out, in, k, k] -> [out]
                    stage_vec = stage_vec + g.detach().abs().mean(dim=(1, 2, 3))
                    used += 1
                if used > 0:
                    task_scores[stage] += stage_vec / used
                    task_counts[stage] += 1

                if feat_cache[stage]:
                    cscore = self._contrast_score_from_feat(feat_cache[stage][-1], labels)
                    contrast_scores[stage] += cscore
                    contrast_counts[stage] += 1

            n_batches += 1

        for h in handles:
            h.remove()

        scores = {}
        for stage in range(num_stages):
            t = task_scores[stage] / max(task_counts[stage], 1)
            c = contrast_scores[stage] / max(contrast_counts[stage], 1)

            def _norm(x):
                x = x - x.min()
                return x / x.max().clamp_min(1e-8)

            I = self.alpha * _norm(t) + (1.0 - self.alpha) * _norm(c)
            scores[f"stage{stage}"] = I.detach().cpu()
            print(
                f"[score] stage{stage} C={widths[stage]} "
                f"I(mean)={I.mean().item():.4f} I(max)={I.max().item():.4f}"
            )
        return scores


def choose_keep_indices(
    scores: Dict[str, torch.Tensor],
    prune_rates: Sequence[float],
    min_keep: int = 4,
) -> Dict[int, torch.Tensor]:
    keep_idx = {}
    for s, rate in enumerate(prune_rates):
        I = scores[f"stage{s}"]
        c = I.numel()
        k = max(min_keep, int(round(c * (1.0 - float(rate)))))
        k = min(k, c)
        topk = torch.topk(I, k, largest=True).indices.sort().values
        keep_idx[s] = topk
        print(f"[keep] stage{s}: keep {k}/{c} (prune_rate={rate})")
    return keep_idx
