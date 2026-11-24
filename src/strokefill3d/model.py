"""Mask-conditioned 3D SwinUNETR reconstruction model."""
from __future__ import annotations

import torch
from monai.networks.nets import SwinUNETR


class StrokeFillNet(torch.nn.Module):
    """Predict a healthy T1 volume from a masked image and its binary mask.

    Unlike diffusion-based LIT, this model is a deterministic, single-pass
    reconstruction network. The mask is an explicit conditioning channel.
    """

    def __init__(self, img_size: tuple[int, int, int] = (96, 96, 96),
                 feature_size: int = 48, use_checkpoint: bool = True):
        super().__init__()
        self.backbone = SwinUNETR(
            in_channels=2,
            out_channels=1,
            spatial_dims=3,
            img_size=img_size,
            feature_size=feature_size,
            use_checkpoint=use_checkpoint,
            use_v2=True,
        )

    def forward(self, masked_t1: torch.Tensor, lesion_mask: torch.Tensor) -> torch.Tensor:
        x = torch.cat((masked_t1, lesion_mask.to(masked_t1.dtype)), dim=1)
        # Bound output to the normalized training range.
        return torch.tanh(self.backbone(x))
