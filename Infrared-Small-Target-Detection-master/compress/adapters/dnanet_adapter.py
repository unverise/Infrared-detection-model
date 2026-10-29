"""
DNANet Adapter
--------------
Maps DNANet into stage-wise prunable groups (shared keep mask per scale).

Stage i <-> nb_filter[i] output width. All Res_CBAM blocks that output Ci
share the same keep indices so nested concat stays semantically aligned.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import torch.nn as nn

from compress.groups import PrunableGroup


# Layer name -> ordered list of input stage ids that form its concat layout.
# None means RGB input (3 channels), not a stage.
DNA_NET_IN_LAYOUT = {
    "conv0_0": [None],
    "conv1_0": [0],
    "conv2_0": [1],
    "conv3_0": [2],
    "conv4_0": [3],
    "conv0_1": [0, 1],
    "conv1_1": [1, 2, 0],
    "conv2_1": [2, 3, 1],
    "conv3_1": [3, 4, 2],
    "conv0_2": [0, 0, 1],
    "conv1_2": [1, 1, 2, 0],
    "conv2_2": [2, 2, 3, 1],
    "conv0_3": [0, 0, 0, 1],
    "conv1_3": [1, 1, 1, 2, 0],
    "conv0_4": [0, 0, 0, 0, 1],
    "conv0_4_final": [0, 0, 0, 0, 0],
    "conv0_4_1x1": [4],
    "conv0_3_1x1": [3],
    "conv0_2_1x1": [2],
    "conv0_1_1x1": [1],
    "final": [0],
    "final1": [0],
    "final2": [0],
    "final3": [0],
    "final4": [0],
}


class DNANetAdapter:
    def __init__(self, nb_filter: Sequence[int]):
        self.nb_filter = list(nb_filter)
        self.num_stages = len(self.nb_filter)

    def enumerate_groups(self, model: nn.Module) -> List[PrunableGroup]:
        groups = []
        for stage, width in enumerate(self.nb_filter):
            module_names = []
            for name, module in model.named_modules():
                if isinstance(module, nn.Conv2d) and module.out_channels == width:
                    # skip SA 7x7 (out=1) and final 1-channel heads
                    if module.out_channels == 1:
                        continue
                    module_names.append(name)
            groups.append(
                PrunableGroup(
                    group_id=f"stage{stage}_width",
                    kind="conv_out",
                    stage_id=stage,
                    channels=width,
                    module_names=module_names,
                )
            )
        return groups

    def stage_of_out_channels(self, out_channels: int) -> int:
        try:
            return self.nb_filter.index(out_channels)
        except ValueError:
            return -1

    def get_hook_modules(self, model: nn.Module) -> Dict[int, List[nn.Module]]:
        """Return Res_CBAM-like blocks (modules that have conv2) grouped by stage."""
        from model.model_DNANet import Res_CBAM_block, VGG_CBAM_Block

        hooks = {s: [] for s in range(self.num_stages)}
        for name, module in model.named_modules():
            if isinstance(module, (Res_CBAM_block, VGG_CBAM_Block)):
                stage = self.stage_of_out_channels(module.conv2.out_channels)
                if stage >= 0:
                    hooks[stage].append(module)
        return hooks

    def get_distill_hooks(self, model: nn.Module) -> List[str]:
        """Feature names useful for optional feature KD (final fusion only for v1)."""
        return ["conv0_4_final"]
