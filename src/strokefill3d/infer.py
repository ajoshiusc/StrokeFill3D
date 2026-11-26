"""Inpaint T1 lesions with EMA weights and observable anatomical conditioning."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from monai.transforms import (
    Compose,
    LoadImage,
    Orientation,
    ScaleIntensityRangePercentiles,
    Spacing,
)
from scipy.ndimage import binary_dilation

from .anatomy import ANATOMY_VERSION, anatomy_features
from .model import RectifiedFlowUNet
from .sampling import integrate


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for flag in ["image", "mask", "checkpoint", "output"]:
        p.add_argument("--" + flag, required=True)
    p.add_argument("--steps", type=int, default=48)
    p.add_argument("--mask-dilation-voxels", type=int, default=0)
    p.add_argument("--roi-size", type=int, nargs=3)
    p.add_argument("--overlap", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--time-embedding-scale", type=float)
    p.add_argument(
        "--strategy", choices=["auto", "centered", "gaussian"], default="auto"
    )
    p.add_argument("--weights", choices=["auto", "ema", "online"], default="auto")
    p.add_argument("--device", choices=["cuda", "cpu"])
    p.add_argument("--amp", action="store_true")
    p.add_argument(
        "--mni",
        action="store_true",
        help="Confirm input is registered to MNI space; required for anatomical checkpoints",
    )
    args = p.parse_args()
    if args.steps < 1 or args.mask_dilation_voxels < 0:
        raise ValueError("Invalid steps or dilation")
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    cfg = ck.get("training_config", {})
    mc = ck.get("model_config", {})
    anatomical = bool(mc.get("anatomy_channels", 0))
    if anatomical and (not args.mni or ck.get("anatomy_version") != ANATOMY_VERSION):
        raise ValueError(
            "Anatomical checkpoints require the matching conditioning version and --mni"
        )
    pre = Compose(
        [
            LoadImage(image_only=True, ensure_channel_first=True),
            Orientation(axcodes="RAS"),
            Spacing(pixdim=(1.0,) * 3, mode="bilinear"),
        ]
    )
    raw_meta = pre(args.image).float()
    raw = raw_meta[0].cpu().numpy()
    if not np.isfinite(raw).all():
        raise ValueError("Input contains nonfinite values")
    lo, hi = np.percentile(raw, [1, 99])
    if hi <= lo:
        raise ValueError("Degenerate input intensity range")
    context = (
        ScaleIntensityRangePercentiles(lower=1, upper=99, b_min=-1, b_max=1, clip=True)(
            raw_meta
        )
        .as_tensor()
        .float()[None]
    )
    mask_meta = Compose(
        [
            LoadImage(image_only=True, ensure_channel_first=True),
            Orientation(axcodes="RAS"),
            Spacing(pixdim=(1.0,) * 3, mode="nearest"),
        ]
    )(args.mask)
    if mask_meta.shape[1:] != raw.shape or not np.allclose(
        mask_meta.affine, raw_meta.affine, atol=1e-4
    ):
        raise ValueError("Image and mask grids differ")
    missing = mask_meta[0].numpy() > 0
    if not missing.any():
        raise ValueError("Empty mask")
    if args.mask_dilation_voxels:
        missing = binary_dilation(missing, iterations=args.mask_dilation_voxels)
    mask = torch.from_numpy(missing[None, None].astype(np.float32))
    anatomy = (
        anatomy_features(context[0, 0].numpy(), missing, raw_meta.affine.numpy())[None]
        if anatomical
        else None
    )
    noise = torch.randn(
        context.shape, generator=torch.Generator(device="cpu").manual_seed(args.seed)
    )
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model = RectifiedFlowUNet(**mc).to(device).eval()
    use_ema = args.weights != "online" and ck.get("ema_model") is not None
    if args.weights == "ema" and not use_ema:
        raise ValueError("Checkpoint has no EMA weights")
    model.load_state_dict(ck["ema_model"] if use_ema else ck["model"])
    roi = tuple(args.roi_size or cfg.get("patch_size", (64,) * 3))
    sample, plan = integrate(
        model,
        context.to(device),
        mask.to(device),
        anatomy.to(device) if anatomy is not None else None,
        noise.to(device),
        roi,
        args.steps,
        args.time_embedding_scale or cfg.get("time_embedding_scale", 1000.0),
        args.strategy,
        args.overlap,
        args.amp,
    )
    fill = (sample[0, 0].clamp(-1, 1).cpu().numpy() + 1) * 0.5 * (hi - lo) + lo
    result = np.where(missing, fill, raw).astype(np.float32)
    if not np.array_equal(result[~missing], raw[~missing]):
        raise RuntimeError("Observed voxel values changed")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    header = nib.load(args.image).header.copy()
    header.set_data_dtype(np.float32)
    nib.save(nib.Nifti1Image(result, raw_meta.affine.numpy(), header), str(output))
    metadata = {
        "checkpoint": args.checkpoint,
        "epoch": ck.get("epoch"),
        "step": ck.get("step"),
        "weights": "ema" if use_ema else "online",
        "roi": roi,
        "steps": args.steps,
        "seed": args.seed,
        "mask_dilation_voxels": args.mask_dilation_voxels,
        "anatomy_version": ck.get("anatomy_version"),
        "outside_mask_max_error": 0.0,
        **plan,
    }
    output.with_name(output.name + ".json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print(json.dumps(metadata), flush=True)


if __name__ == "__main__":
    main()
