"""Utilities for the controlled real-contour conditional rectified-flow study."""

from __future__ import annotations

import torch
from monai.networks.nets import DiffusionModelUNet


def make_model(config: dict) -> DiffusionModelUNet:
    return DiffusionModelUNet(**config)


def coords_from_starts(starts: torch.Tensor) -> torch.Tensor:
    axis = torch.arange(64, device=starts.device, dtype=torch.float32)
    xx, zz = torch.meshgrid(axis, axis, indexing="ij")
    return torch.stack(
        [
            (starts[:, 0, None, None] + xx - 98) / 100,
            starts[:, 1, None, None].expand(-1, 64, 64) / 100,
            (starts[:, 2, None, None] + zz - 72) / 100,
        ],
        dim=1,
    )


def field(model, state, image, mask, coords, time):
    if not torch.is_tensor(time):
        time = torch.full((len(state),), float(time), device=state.device)
    with torch.autocast(state.device.type, dtype=torch.float16, enabled=state.is_cuda):
        pred = model(torch.cat([state, image, mask, coords], 1), time * 1000).float()
    return (pred - state) / (1 - time[:, None, None, None]).clamp_min(0.05)


@torch.no_grad()
def sample(model, image, mask, coords, seed=909, steps=20, method="heun_endpoint"):
    """RF sampling; integrate only over the trained [0,.95] interval.

    At t=.95 the clean-MRI head provides the endpoint estimate directly. The
    previous sampler extrapolated the last velocity from .95 to 1.0.
    """
    if steps < 2 or method not in {"heun_endpoint", "euler_endpoint", "legacy_euler"}:
        raise ValueError("Unsupported sampling protocol")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    noise = torch.randn(image.shape, generator=generator).to(image.device)
    state = torch.where(mask > 0, noise, image)
    if method == "legacy_euler":
        dt = 1.0 / steps
        for i in range(steps):
            t = i / steps
            state = torch.where(
                mask > 0,
                state + dt * field(model, state, image, mask, coords, t),
                image,
            )
        return torch.where(mask > 0, state.clamp(-1, 1), image)

    t_max = 0.95
    dt = t_max / steps
    for i in range(steps):
        t = i * dt
        v1 = field(model, state, image, mask, coords, t)
        proposal = torch.where(mask > 0, state + dt * v1, image)
        if method == "heun_endpoint" and i < steps - 1:
            v2 = field(model, proposal, image, mask, coords, t + dt)
            state = torch.where(mask > 0, state + 0.5 * dt * (v1 + v2), image)
        else:
            state = proposal
    endpoint_v = field(model, state, image, mask, coords, t_max)
    clean = state + (1 - t_max) * endpoint_v
    clean = torch.where(mask > 0, clean.clamp(-1, 1), image)
    if not torch.isfinite(clean).all() or not torch.equal(
        clean[mask == 0], image[mask == 0]
    ):
        raise RuntimeError("Invalid RF sample or changed visible voxels")
    return clean


def masked_metrics(
    pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> dict:
    m = mask.float()
    denom = m.flatten(1).sum(1).clamp_min(1)
    mae = ((pred - target).abs() * m).flatten(1).sum(1) / denom
    gx_p = pred[..., 1:] - pred[..., :-1]
    gx_t = target[..., 1:] - target[..., :-1]
    gy_p = pred[..., 1:, :] - pred[..., :-1, :]
    gy_t = target[..., 1:, :] - target[..., :-1, :]
    mx = m[..., 1:] * m[..., :-1]
    my = m[..., 1:, :] * m[..., :-1, :]
    grad_num = ((gx_p - gx_t).abs() * mx).flatten(1).sum(1)
    grad_num += ((gy_p - gy_t).abs() * my).flatten(1).sum(1)
    grad_den = mx.flatten(1).sum(1) + my.flatten(1).sum(1)
    # Local SSIM, averaged only over the masked region.
    import torch.nn.functional as F

    ux = F.avg_pool2d(pred, 7, 1, 3)
    uy = F.avg_pool2d(target, 7, 1, 3)
    vx = (F.avg_pool2d(pred.square(), 7, 1, 3) - ux.square()).clamp_min(0)
    vy = (F.avg_pool2d(target.square(), 7, 1, 3) - uy.square()).clamp_min(0)
    cov = F.avg_pool2d(pred * target, 7, 1, 3) - ux * uy
    c1, c2 = 0.0004, 0.0036
    ssim_map = ((2 * ux * uy + c1) * (2 * cov + c2)) / (
        (ux.square() + uy.square() + c1) * (vx + vy + c2)
    )
    ssim = (ssim_map * m).flatten(1).sum(1) / denom
    return {
        "masked_mae": float(mae.mean()),
        "masked_gradient_mae": float((grad_num / grad_den.clamp_min(1)).mean()),
        "masked_ssim": float(ssim.mean()),
    }
