"""Shared centered/Gaussian flow integration for checkpoint-compatible inference."""

import numpy as np
import torch
from monai.data.utils import compute_importance_map, dense_patch_slices
from monai.inferers.utils import _get_scan_interval


def window_plan(mask, roi, strategy="auto", overlap=0.5, margin=8):
    if not 0 <= overlap < 1:
        raise ValueError("overlap must lie in [0,1)")
    shape = np.asarray(mask.shape)
    roi = np.asarray(roi)
    if (shape < roi).any():
        raise ValueError("Input must be at least the checkpoint ROI size on each axis")
    coords = np.argwhere(mask)
    if not len(coords):
        raise ValueError("Empty inpainting mask")
    low = coords.min(0)
    high = coords.max(0) + 1
    starts = np.clip((low + high - roi) // 2, 0, shape - roi)
    fits = bool(
        np.all(low - starts >= margin) and np.all(starts + roi - high >= margin)
    )
    if strategy == "centered" and not fits:
        raise ValueError(
            "Lesion does not fit the centered ROI with context; use gaussian"
        )
    if strategy == "centered" or (strategy == "auto" and fits):
        return [
            tuple(slice(int(s), int(s + n)) for s, n in zip(starts, roi))
        ], "centered"
    windows = dense_patch_slices(
        tuple(shape),
        tuple(roi),
        _get_scan_interval(
            tuple(shape), tuple(roi), len(shape), (overlap,) * len(shape)
        ),
    )
    return [s for s in windows if mask[s].any()], "gaussian"


@torch.inference_mode()
def integrate(
    model,
    context,
    mask,
    anatomy,
    noise,
    roi,
    steps=48,
    time_scale=1000.0,
    strategy="auto",
    overlap=0.5,
    amp=False,
):
    if steps < 1:
        raise ValueError("steps must be positive")
    windows, chosen = window_plan(mask[0, 0].cpu().numpy() > 0, roi, strategy, overlap)
    weight = (
        torch.ones(tuple(roi), device=context.device)
        if chosen == "centered"
        else compute_importance_map(
            tuple(roi), mode="gaussian", sigma_scale=0.125, device=context.device
        )
    )
    denominator = torch.zeros_like(mask)
    for window in windows:
        denominator[(slice(None), slice(None), *window)] += weight
    if not torch.all(denominator[mask > 0] > 0):
        raise RuntimeError("Uncovered lesion voxels")
    masked = torch.where(mask > 0, torch.zeros_like(context), context)
    sample = torch.where(mask > 0, noise, context).clone()
    for step in range(steps):
        channels = [sample, masked, mask]
        if anatomy is not None:
            channels.append(anatomy)
        inputs = torch.cat(channels, dim=1)
        velocity = torch.zeros_like(sample)
        for window in windows:
            sl = (slice(None), slice(None), *window)
            times = torch.full((1,), step / steps * time_scale, device=context.device)
            with torch.autocast(
                device_type=context.device.type,
                dtype=torch.float16,
                enabled=context.device.type == "cuda" and amp,
            ):
                prediction = model.predict_input(inputs[sl], times)
            velocity[sl] += prediction.float() * weight
        sample = torch.where(
            mask > 0, sample + (velocity / denominator.clamp_min(1e-8)) / steps, context
        )
        if not torch.isfinite(sample).all():
            raise RuntimeError("Nonfinite inference state")
    return sample, {"strategy": chosen, "windows_per_step": len(windows)}
