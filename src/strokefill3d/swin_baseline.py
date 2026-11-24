"""Optional deterministic baseline retained for comparison experiments."""
from __future__ import annotations
import torch
from monai.networks.nets import SwinUNETR


class StrokeFillSwinBaseline(torch.nn.Module):
    def __init__(self, img_size=(48, 48, 48), feature_size=24, use_checkpoint=True):
        super().__init__()
        self.backbone = SwinUNETR(
            in_channels=2, out_channels=1, spatial_dims=3, img_size=img_size,
            feature_size=feature_size, use_checkpoint=use_checkpoint, use_v2=True,
        )

    def forward(self, masked_t1, lesion_mask):
        return torch.tanh(self.backbone(torch.cat((masked_t1, lesion_mask.to(masked_t1.dtype)), dim=1)))
