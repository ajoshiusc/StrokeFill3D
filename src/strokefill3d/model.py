"""MONAI 3D U-Net backbone for conditional rectified-flow inpainting."""
from __future__ import annotations

import torch
from monai.networks.nets import DiffusionModelUNet


class RectifiedFlowUNet(torch.nn.Module):
    """Predict the velocity from Gaussian noise to healthy T1 tissue.

    Inputs are the evolving sample, masked T1 context, and a binary lesion mask.
    The network predicts only a velocity field; the ODE integration is in infer.py.
    """

    def __init__(self, channels=(32, 64, 128), attention_levels=(False, True, True),
                 num_head_channels=(0, 32, 32), num_res_blocks=1):
        super().__init__()
        self.backbone = DiffusionModelUNet(
            spatial_dims=3,
            in_channels=3,
            out_channels=1,
            num_res_blocks=num_res_blocks,
            channels=tuple(channels),
            attention_levels=tuple(attention_levels),
            num_head_channels=tuple(num_head_channels),
            norm_num_groups=8,
            norm_eps=1e-6,
        )

    def forward(self, sample: torch.Tensor, context: torch.Tensor,
                lesion_mask: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        x = torch.cat((sample, context, lesion_mask.to(sample.dtype)), dim=1)
        return self.backbone(x=x, timesteps=timesteps)
