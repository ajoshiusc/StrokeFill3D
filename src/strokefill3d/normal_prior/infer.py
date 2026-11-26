"""Centered 96-cube normal-prior RF inference on registered MNI NIfTI data."""

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from scipy.ndimage import binary_dilation, label

from .core import coordinate_maps, fit_source, load_bundle
from .data import check_grid, normalize
from .io import atomic_json


def centered_roi(missing, shape, size, margin=8):
    loc = np.argwhere(missing)
    if not len(loc):
        raise ValueError("Empty lesion component")
    size = np.array(size)
    shape = np.array(shape)
    if (shape < size).any():
        raise ValueError("Image field of view is smaller than the trained patch")
    lo = loc.min(0)
    hi = loc.max(0) + 1
    start = np.clip((lo + hi - size) // 2, 0, shape - size)
    if ((lo - start) < margin).any() or ((start + size - hi) < margin).any():
        raise ValueError(
            "Lesion component does not fit the trained 96-cube with visible context. No silent resizing or Gaussian averaging is performed"
        )
    return start, tuple(slice(int(s), int(s + n)) for s, n in zip(start, size))


def generate_nifti(
    model, codec, image_path, mask_path, output, device, settings, metadata=None
):
    nii = nib.load(image_path)
    masknii = nib.load(mask_path)
    check_grid(nii, masknii)
    raw = np.asarray(nii.dataobj, dtype=np.float32)
    missing = np.asarray(masknii.dataobj) > 0
    if settings.get("dilation", 0):
        missing = binary_dilation(missing, iterations=settings["dilation"])
    if not missing.any():
        raise ValueError("Empty lesion mask")
    normalized, lo, hi = normalize(raw, missing)
    components, n = label(missing)
    result = normalized.copy()
    remaining = missing.copy()
    records = []
    # Plan every component before generation; oversized masks fail before partial work.
    plans = [
        centered_roi(
            components == i, raw.shape, codec.patch_size, settings.get("margin", 8)
        )
        for i in range(1, n + 1)
    ]
    for i, (starts, sl) in enumerate(plans, 1):
        local_mask = remaining[sl]
        context = np.where(local_mask, 0, result[sl]).astype(np.float32)
        image = torch.from_numpy(context[None, None]).to(device)
        mask = torch.from_numpy(local_mask[None, None].astype(np.float32)).to(device)
        origin = nii.affine[:3, :3] @ starts + nii.affine[:3, 3]
        coords = coordinate_maps(
            torch.tensor(origin, dtype=torch.float32, device=device)[None],
            [v // codec.factor for v in codec.patch_size],
            codec.factor,
        )
        pred, _, _, history = fit_source(
            model,
            codec,
            image,
            mask,
            coords,
            seed=settings["seed"] + i - 1,
            steps=settings["flow_steps"],
            iterations=settings["iterations"],
            trust_weight=settings["trust_weight"],
            boundary_weight=settings["boundary_weight"],
            checkpoint_steps=True,
            callback=lambda r, component=i: print(
                f"component={component}/{n} " + json.dumps(r), flush=True
            ),
        )
        owned = components[sl] == i
        crop = result[sl]
        crop[owned] = pred[0, 0].cpu().numpy()[owned]
        remaining[components == i] = False
        records.append(
            {
                "component": i,
                "starts": starts.tolist(),
                "origin": origin.tolist(),
                "history": history,
            }
        )
    fill = (result + 1) * 0.5 * (hi - lo) + lo
    output_data = np.where(missing, fill, raw).astype(np.float32)
    if not np.isfinite(output_data).all() or not np.array_equal(
        output_data[~missing], raw[~missing]
    ):
        raise RuntimeError(
            "Output changed observed voxels or contains nonfinite values"
        )
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    header = nii.header.copy()
    header.set_data_dtype(np.float32)
    nib.save(nib.Nifti1Image(output_data, nii.affine, header), output)
    atomic_json(
        dict(
            **(metadata or {}),
            settings=settings,
            components=records,
            outside_mask_max_error=0.0,
            normalization_percentiles=[lo, hi],
            normalization_uses_observed_voxels_only=True,
        ),
        str(output) + ".json",
    )
    return output_data


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ["run", "image", "mask", "output"]:
        p.add_argument("--" + key, required=True)
    p.add_argument("--checkpoint", default="best.pt")
    p.add_argument("--mni", action="store_true", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--flow-steps", type=int, default=16)
    p.add_argument("--iterations", type=int, default=120)
    p.add_argument("--seed", type=int, default=909)
    p.add_argument("--mask-dilation-voxels", type=int, default=2)
    args = p.parse_args()
    if args.mask_dilation_voxels < 0:
        raise ValueError("Negative dilation")
    model, codec, saved = load_bundle(args.run, args.checkpoint, args.device)
    generate_nifti(
        model,
        codec,
        args.image,
        args.mask,
        args.output,
        torch.device(args.device),
        {
            "seed": args.seed,
            "flow_steps": args.flow_steps,
            "iterations": args.iterations,
            "trust_weight": 0.001,
            "boundary_weight": 4.0,
            "dilation": args.mask_dilation_voxels,
            "margin": 8,
        },
        {
            "checkpoint_step": saved["step"],
            "codec_sha256": saved["codec_sha256"],
            "run": str(args.run),
        },
    )


if __name__ == "__main__":
    main()
