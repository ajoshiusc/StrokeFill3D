"""3D extension of the tested conditional clean-MRI rectified-flow model."""

import torch

from strokefill3d.normal_prior.core import amp, coordinate_maps


def velocity(model, state, image, mask, coords, time):
    if not torch.is_tensor(time):
        time = torch.full((len(state),), float(time), device=state.device)
    with amp(state.device):
        clean = model(torch.cat([state, image, mask, coords], 1), time * 1000).float()
    return (clean - state) / (1 - time[:, None, None, None, None]).clamp_min(0.05)


@torch.no_grad()
def sample(model, image, mask, origin, seed=909, steps=20):
    if steps != 20:
        raise ValueError("Euler20 matches training t in [0,.95)")
    coords = coordinate_maps(origin, image.shape[2:], 1)
    noise = torch.randn(image.shape, generator=torch.Generator().manual_seed(seed)).to(
        image.device
    )
    state = torch.where(mask > 0, noise, image)
    for i in range(steps):
        state = torch.where(
            mask > 0,
            state + velocity(model, state, image, mask, coords, i / steps) / steps,
            image,
        )
    out = torch.where(mask > 0, state.clamp(-1, 1), image)
    if not torch.isfinite(out).all() or not torch.equal(
        out[mask == 0], image[mask == 0]
    ):
        raise RuntimeError("Invalid RF sample")
    return out
