"""3D codec, complete-code RF prior and observation-only source fitting."""

import hashlib
from pathlib import Path

import torch
import torch.nn.functional as F
from monai.networks.nets import AutoencoderKL, DiffusionModelUNet
from torch.utils.checkpoint import checkpoint

from strokefill3d.model import checkpoint_blocks

VERSION = "normal_prior_3d_v1"


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_checkpoint_snapshot(path, expected_sha256=None):
    """Hash and load the same open file across atomic training-checkpoint updates."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
        if expected_sha256 is not None and h.hexdigest() != expected_sha256:
            raise ValueError("Checkpoint fingerprints differ")
        f.seek(0)
        saved = torch.load(f, map_location="cpu", weights_only=True)
    return saved, h.hexdigest()


def amp(device):
    return torch.autocast(
        device.type, dtype=torch.float16, enabled=device.type == "cuda"
    )


def make_autoencoder(config):
    if config.get("spatial_dims") != 3:
        raise ValueError("A separately trained 3D autoencoder is required")
    return AutoencoderKL(**config)


def make_prior(config):
    config = dict(config)
    activation = config.pop("activation_checkpointing", False)
    if config.get("spatial_dims") != 3:
        raise ValueError(
            "Expected a 3D RF model; 2D/legacy checkpoints are incompatible"
        )
    model = DiffusionModelUNet(**config)
    if activation:
        checkpoint_blocks(model)
    return model


def prior_config(config, channels):
    result = dict(config)
    result.update(spatial_dims=3, in_channels=channels + 3, out_channels=channels)
    return result


def transform_channels(tensor, matrix):
    return torch.einsum("bc...,cd->bd...", tensor.float(), matrix.float())


class Codec(torch.nn.Module):
    def __init__(self, saved):
        super().__init__()
        if saved.get("kind") != VERSION + "_codec":
            raise ValueError("Not a normal-prior 3D codec")
        self.config = saved["ae_config"]
        self.ae = make_autoencoder(self.config)
        self.ae.load_state_dict(saved["model"])
        self.patch_size = tuple(saved["patch_size"])
        self.factor = int(saved["factor"])
        self.channels = int(saved["whiten"].shape[1])
        self.quality = saved.get("quality", {})
        for key in ["mean", "std", "whiten", "unwhiten"]:
            self.register_buffer(key, saved[key].float())
        self.eval().requires_grad_(False)

    @torch.no_grad()
    def encode(self, image):
        with amp(image.device):
            mu, _ = self.ae.encode(image.float())
        return transform_channels((mu.float() - self.mean) / self.std, self.whiten)

    def decode(self, code):
        mu = transform_channels(code, self.unwhiten) * self.std + self.mean
        with amp(code.device):
            return self.ae.decode(mu).float()


def coordinate_maps(origins, shape, factor):
    axes = [
        torch.arange(n, device=origins.device, dtype=torch.float32) * factor
        + (factor - 1) / 2
        for n in shape
    ]
    grid = torch.stack(torch.meshgrid(*axes, indexing="ij"))
    return (grid[None] + origins[:, :, None, None, None]) / 100


def velocity(model, state, coords, time):
    if not torch.is_tensor(time):
        time = torch.full((len(state),), float(time), device=state.device)
    with amp(state.device):
        return model(torch.cat([state, coords], 1), time * 1000).float()


def solve(model, source, coords, steps, checkpoint_steps=True):
    if steps < 1:
        raise ValueError("Flow steps must be positive")
    state = source
    for i in range(steps):
        t = i / steps

        def call(x, time=t):
            return velocity(model, x, coords, time)

        v = (
            checkpoint(call, state, use_reentrant=False, preserve_rng_state=False)
            if checkpoint_steps and state.requires_grad
            else call(state)
        )
        state = state + v / steps
    return state


def validate_observations(image, mask, coords, codec):
    if (
        image.ndim != 5
        or image.shape != mask.shape
        or image.shape[1] != 1
        or tuple(image.shape[2:]) != codec.patch_size
    ):
        raise ValueError(
            f"Expected Bx1x{codec.patch_size} MRI/mask; 2D models are not supported"
        )
    expected = (len(image), 3, *[n // codec.factor for n in codec.patch_size])
    if tuple(coords.shape) != expected:
        raise ValueError(f"Wrong MNI coordinate shape: expected {expected}")
    if not all(torch.isfinite(v).all() for v in [image, mask, coords]):
        raise ValueError("Nonfinite input")
    if torch.any((mask != 0) & (mask != 1)):
        raise ValueError("Mask must be binary")
    if torch.any(image[mask > 0] != 0):
        raise ValueError("Hidden MRI intensities must be zero")


def fit_source(
    model,
    codec,
    image,
    mask,
    coords,
    *,
    seed=909,
    steps=16,
    iterations=120,
    learning_rate=0.05,
    trust_weight=0.001,
    boundary_weight=4.0,
    checkpoint_steps=True,
    callback=None,
):
    """D-Flow-style inverse fit. No target argument and no hidden-intensity loss."""
    validate_observations(image, mask, coords, codec)
    if iterations < 0 or steps < 1:
        raise ValueError("Invalid source optimization settings")
    model.eval().requires_grad_(False)
    shape = (len(image), codec.channels, *coords.shape[2:])
    initial = torch.randn(shape, generator=torch.Generator().manual_seed(seed)).to(
        image.device
    )
    source = torch.nn.Parameter(initial.clone())
    optimizer = torch.optim.Adam([source], lr=learning_rate)
    scaler = torch.amp.GradScaler(
        image.device.type, enabled=image.is_cuda, init_scale=1024
    )
    known = 1 - mask
    rim = (F.max_pool3d(mask, 9, 1, 4) - mask).clamp(0, 1) * known
    weight = known * (1 + 2 * (image > -0.95) + boundary_weight * rim)

    def weighted(x, w):
        return ((x * w).flatten(1).sum(1) / w.flatten(1).sum(1).clamp_min(1)).mean()

    history = []
    for iteration in range(iterations + 1):
        optimizer.zero_grad(set_to_none=True)
        code = solve(model, source, coords, steps, checkpoint_steps)
        raw = codec.decode(code)
        measured = weighted((raw - image).square(), weight)
        edge = 0
        for dim in (2, 3, 4):
            pair = known.narrow(dim, 0, known.shape[dim] - 1) * known.narrow(
                dim, 1, known.shape[dim] - 1
            )
            edge = (
                edge
                + weighted((raw.diff(dim=dim) - image.diff(dim=dim)).square(), pair) / 3
            )
        trust = (source - initial).square().mean()
        loss = measured + 0.25 * edge + trust_weight * trust
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite source-fitting loss")
        if iteration % 20 == 0 or iteration == iterations:
            row = {
                "iteration": iteration,
                "observed_mse": float(measured.detach()),
                "observed_gradient_mse": float(edge.detach()),
                "source_change_mse": float(trust.detach()),
            }
            history.append(row)
            if callback:
                callback(row)
        if iteration == iterations:
            break
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_([source], 1)
        scaler.step(optimizer)
        scaler.update()
    result = torch.where(mask > 0, raw.detach().clamp(-1, 1), image)
    if not torch.isfinite(result).all() or not torch.equal(
        result[mask == 0], image[mask == 0]
    ):
        raise RuntimeError("Invalid generated image")
    return result, code.detach(), source.detach(), history


def load_bundle(run, checkpoint_name="best.pt", device="cuda", allow_functional=False):
    run = Path(run)
    path = Path(checkpoint_name)
    if not path.is_absolute():
        path = run / "prior" / path
    codec_path = run / "codec.pt"
    saved, checkpoint_sha = load_checkpoint_snapshot(path)
    if saved.get("kind") != VERSION + "_prior":
        raise ValueError("Not a normal-prior 3D RF checkpoint")
    codec_state, _ = load_checkpoint_snapshot(
        codec_path, expected_sha256=saved["codec_sha256"]
    )
    codec = Codec(codec_state).to(device)
    if not codec.quality.get("passed", False) and not allow_functional:
        raise ValueError("The codec has not passed its reconstruction gate")
    model = make_prior(saved["model_config"]).to(device).eval()
    model.load_state_dict(saved["model"])
    if saved["model_config"]["out_channels"] != codec.channels:
        raise ValueError("Code channel mismatch")
    saved["loaded_checkpoint_sha256"] = checkpoint_sha
    return model, codec, saved
