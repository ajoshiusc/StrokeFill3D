"""MONAI 3D U-Net backbones for conditional rectified flow."""

from __future__ import annotations

import torch
from monai.networks.blocks import SABlock
from monai.networks.nets import DiffusionModelUNet
from monai.networks.nets.diffusion_model_unet import DiffusionUNetResnetBlock
from torch.utils.checkpoint import checkpoint


class CheckpointBlock(torch.nn.Module):
    def __init__(self, block):
        super().__init__()
        self.block = block

    def forward(self, *args, **kwargs):
        if self.training and torch.is_grad_enabled():
            return checkpoint(self.block, *args, use_reentrant=False, **kwargs)
        return self.block(*args, **kwargs)


def checkpoint_blocks(module):
    # Recompute small blocks individually; wrapping the whole U-Net can retain a
    # complete recomputed graph and lose much of the intended memory saving.
    for name, child in list(module.named_children()):
        if isinstance(child, (DiffusionUNetResnetBlock, SABlock)):
            setattr(module, name, CheckpointBlock(child))
        else:
            checkpoint_blocks(child)


class RectifiedFlowUNet(torch.nn.Module):
    def __init__(
        self,
        channels=(32, 64, 128),
        attention_levels=(False, False, True),
        num_head_channels=(0, 0, 32),
        num_res_blocks=1,
        anatomy_channels=0,
        use_flash_attention=False,
        activation_checkpointing=False,
    ):
        super().__init__()
        self.anatomy_channels = int(anatomy_channels)
        self.activation_checkpointing = activation_checkpointing
        self.backbone = DiffusionModelUNet(
            spatial_dims=3,
            in_channels=3 + self.anatomy_channels,
            out_channels=1,
            num_res_blocks=num_res_blocks,
            channels=tuple(channels),
            attention_levels=tuple(attention_levels),
            num_head_channels=tuple(num_head_channels),
            norm_num_groups=8,
            norm_eps=1e-6,
            use_flash_attention=use_flash_attention,
        )
        if activation_checkpointing:
            checkpoint_blocks(self.backbone)

    def predict_input(self, x, timesteps):
        return self.backbone(x=x, timesteps=timesteps)

    def forward(self, sample, context, lesion_mask, timesteps, anatomy=None):
        parts = [sample, context, lesion_mask.to(sample.dtype)]
        if self.anatomy_channels:
            if anatomy is None or anatomy.shape[1] != self.anatomy_channels:
                raise ValueError(
                    f"Model requires {self.anatomy_channels} anatomical channels"
                )
            parts.append(anatomy.to(sample.dtype))
        return self.predict_input(torch.cat(parts, dim=1), timesteps)


def model_config(cfg):
    keys = (
        "channels",
        "attention_levels",
        "num_head_channels",
        "num_res_blocks",
        "anatomy_channels",
        "use_flash_attention",
        "activation_checkpointing",
    )
    return {k: cfg[k] for k in keys if k in cfg}
