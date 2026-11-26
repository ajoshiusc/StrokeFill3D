"""Healthy-code rectified flow and visible-data constrained sampling."""

from pathlib import Path
import torch
import torch.nn.functional as F
from monai.networks.nets import AutoencoderKL, DiffusionModelUNet


class FrozenCodec(torch.nn.Module):
    def __init__(self, checkpoint):
        super().__init__()
        self.config = checkpoint["config"]
        if self.config.get("spatial_dims") != 2:
            raise ValueError("This experimental codec supports 2D 64x64 MRI only")
        self.ae = AutoencoderKL(**self.config)
        self.ae.load_state_dict(checkpoint["model"])
        for key in ["mean", "std", "center", "whiten", "unwhiten"]:
            self.register_buffer(key, checkpoint[key].float())
        self.eval().requires_grad_(False)

    @torch.no_grad()
    def encode(self, image):
        with torch.autocast(
            image.device.type, dtype=torch.float16, enabled=image.is_cuda
        ):
            mu, _ = self.ae.encode(image.float())
        raw = (mu.float() - self.mean) / self.std - self.center
        return torch.einsum("bchw,cd->bdhw", raw, self.whiten)

    def decode(self, code):
        raw = torch.einsum("bchw,cd->bdhw", code.float(), self.unwhiten)
        mu = (raw + self.center) * self.std + self.mean
        with torch.autocast(
            code.device.type, dtype=torch.float16, enabled=code.is_cuda
        ):
            return self.ae.decode(mu).float()


MODEL_CONFIG = dict(
    spatial_dims=2,
    in_channels=10,
    out_channels=7,
    channels=(32, 64, 96),
    num_res_blocks=2,
    attention_levels=(True, True, True),
    num_head_channels=(16, 32, 32),
    norm_num_groups=8,
    use_flash_attention=True,
)


def make_model(config=None):
    return DiffusionModelUNet(**(MODEL_CONFIG if config is None else config))


def coordinate_maps(starts, size=16):
    """Starts contain image-grid x, world y, image-grid z; voxel centers match avg-pool(4)."""
    axis = torch.arange(size, device=starts.device, dtype=torch.float32) * 4 + 1.5
    xx, zz = torch.meshgrid(axis, axis, indexing="ij")
    return torch.stack(
        [
            (starts[:, 0, None, None] + xx - 98) / 100,
            starts[:, 1, None, None].expand(-1, size, size) / 100,
            (starts[:, 2, None, None] + zz - 72) / 100,
        ],
        dim=1,
    )


def predict_velocity(model, state, coords, time):
    if not torch.is_tensor(time):
        time = torch.full((len(state),), float(time), device=state.device)
    with torch.autocast(state.device.type, dtype=torch.float16, enabled=state.is_cuda):
        return model(torch.cat([state, coords], dim=1), time * 1000).float()


def validate_observations(image, mask, coords):
    if image.shape != mask.shape or image.ndim != 4 or image.shape[1:] != (1, 64, 64):
        raise ValueError(
            "This 2D prototype expects matching Bx1x64x64 image/mask; 96-cube inference needs a separately trained 3D model"
        )
    if coords.shape != (len(image), 3, 16, 16):
        raise ValueError("Expected Bx3x16x16 MNI coordinate maps")
    if (
        not torch.isfinite(image).all()
        or not torch.isfinite(mask).all()
        or not torch.isfinite(coords).all()
    ):
        raise ValueError("Nonfinite input")
    if torch.any((mask != 0) & (mask != 1)):
        raise ValueError("Inpainting mask must be binary")
    if torch.any(image[mask > 0] != 0):
        raise ValueError("Image must be zero under the inpainting mask")


@torch.inference_mode()
def healthy_sample(model, coords, seed=909, steps=40):
    noise = torch.randn(
        (len(coords), 7, 16, 16), generator=torch.Generator().manual_seed(seed)
    )
    state = noise.to(coords.device)
    for i in range(steps):
        v = predict_velocity(model, state, coords, i / steps)
        trial = state + v / steps
        if i + 1 < steps:
            vn = predict_velocity(model, trial, coords, (i + 1) / steps)
            state = state + (v + vn) / (2 * steps)
        else:
            state = trial
    return state


@torch.inference_mode()
def inpaint(
    model,
    codec,
    image,
    mask,
    coords,
    seed=909,
    steps=40,
    projection=1.0,
    projection_mode="difference",
    return_code=False,
):
    """No target argument: projection uses observed pixels and model estimates only.

    The normal-MRI RF supplies a clean-endpoint estimate. Encoding that image with
    the observed pixels reinserted gives an endpoint data-consistency correction.
    This is a constrained RF sampler, not a ground-truth latent reconstruction.
    """
    validate_observations(image, mask, coords)
    noise = torch.randn(
        (len(image), 7, 16, 16), generator=torch.Generator().manual_seed(seed)
    )
    state = noise.to(image.device)
    for i in range(steps):
        time = i / steps
        v = predict_velocity(model, state, coords, time)
        if projection:
            endpoint = state + (1 - time) * v
            predicted = codec.decode(endpoint).clamp(-1, 1)
            composite = torch.where(mask > 0, predicted, image)
            constrained = codec.encode(composite)
            if projection_mode == "difference":
                # Cancel the AE round-trip error. When there are no observations
                # (or they already match), conditioning must not alter the prior.
                correction = constrained - codec.encode(predicted)
            elif projection_mode == "roundtrip":
                correction = constrained - endpoint
            else:
                raise ValueError("Unknown projection mode")
            v = v + projection * correction / (1 - time)
        state = state + v / steps
    predicted = codec.decode(state).clamp(-1, 1)
    output = torch.where(mask > 0, predicted, image)
    return (output, state) if return_code else output


def load_run(directory, checkpoint="best.pt", device="cuda"):
    directory = Path(directory)
    codec = FrozenCodec(
        torch.load(directory / "codec.pt", map_location="cpu", weights_only=True)
    ).to(device)
    saved = torch.load(directory / checkpoint, map_location="cpu", weights_only=True)
    model = make_model(saved["model_config"]).to(device).eval()
    model.load_state_dict(saved["model"])
    return model, codec, saved


def solve_from_source(model, source, coords, steps=16):
    """Differentiable Euler RF integration; no image-space edits along the path."""
    state = source
    for i in range(steps):
        state = state + predict_velocity(model, state, coords, i / steps) / steps
    return state


def fit_source(
    model,
    codec,
    image,
    mask,
    coords,
    seed=909,
    steps=16,
    iterations=80,
    learning_rate=0.05,
    trust_weight=0.01,
    return_source=False,
    boundary_weight=0.0,
):
    """Fit only the Gaussian source to observed MRI through a frozen RF trajectory.

    This is regularized D-Flow-style source optimization, not exact posterior
    sampling. The loss never reads any intensity under the inpainting mask.
    """
    validate_observations(image, mask, coords)
    model.eval().requires_grad_(False)
    initial = torch.randn(
        (len(image), 7, 16, 16), generator=torch.Generator().manual_seed(seed)
    ).to(image.device)
    source = torch.nn.Parameter(initial.clone())
    optimizer = torch.optim.Adam([source], lr=learning_rate)
    scaler = torch.amp.GradScaler(
        image.device.type, enabled=image.is_cuda, init_scale=1024
    )
    known = 1 - mask
    rim = (F.max_pool2d(mask, 9, stride=1, padding=4) - mask).clamp(0, 1) * known
    observed_weight = known * (1 + 2 * (image > -0.95) + boundary_weight * rim)

    def per_image_mean(value, weight):
        return (value * weight).flatten(1).sum(1) / weight.flatten(1).sum(1).clamp_min(
            1
        )

    history = []
    for iteration in range(iterations + 1):
        optimizer.zero_grad(set_to_none=True)
        code = solve_from_source(model, source, coords, steps)
        predicted = codec.decode(code)
        measured = per_image_mean((predicted - image).square(), observed_weight).mean()
        edge = 0
        for dim in (2, 3):
            pair = known.narrow(dim, 0, 63) * known.narrow(dim, 1, 63)
            edge = (
                edge
                + per_image_mean(
                    (predicted.diff(dim=dim) - image.diff(dim=dim)).square(), pair
                ).mean()
                / 2
            )
        trust = (source - initial).square().mean()
        loss = measured + 0.25 * edge + trust_weight * trust
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite source objective")
        if iteration % 20 == 0 or iteration == iterations:
            history.append(
                dict(
                    iteration=iteration,
                    observed_mse=float(measured.detach()),
                    observed_gradient_mse=float(edge.detach()),
                    source_change_mse=float(trust.detach()),
                )
            )
        if iteration == iterations:
            break
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_([source], 1)
        scaler.step(optimizer)
        scaler.update()
    output = torch.where(mask > 0, predicted.detach().clamp(-1, 1), image)
    result = (output, code.detach(), history)
    return (*result, source.detach()) if return_source else result
