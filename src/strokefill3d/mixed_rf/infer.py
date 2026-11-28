"""Direct conditional RF inpainting of a prepared MNI-space T1 case."""

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from scipy.ndimage import binary_dilation, distance_transform_edt

from strokefill3d.normal_prior.core import make_prior
from strokefill3d.normal_prior.data import check_grid
from strokefill3d.normal_prior.external_infer import extract_patch, plan_tiles
from strokefill3d.normal_prior.io import atomic_json

from .core import sample


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prepared", type=Path, required=True)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=909)
    p.add_argument("--dilation", type=int, default=2)
    a = p.parse_args()
    if a.dilation < 0:
        raise ValueError("Dilation must be nonnegative")
    prep = a.prepared / "preprocessing.json"
    if prep.exists():
        qc = json.loads(prep.read_text())["qc"]
        if (
            qc["brain_mask_dice_with_template"] < 0.8
            or qc["nonpositive_jacobian_voxels_in_brain"]
        ):
            raise ValueError("Preprocessing QC failed")
    nii = nib.load(a.prepared / "t1_brain_training.nii.gz")
    ln = nib.load(a.prepared / "lesion_training.nii.gz")
    bn = nib.load(a.prepared / "brain_mask_training.nii.gz")
    check_grid(nii, ln)
    check_grid(nii, bn)
    if nii.shape != (197, 233, 189) or not np.allclose(
        nii.affine[:3, 3], [-98, -134, -72], atol=1e-3
    ):
        raise ValueError("Expected canonical MNI grid")
    raw = np.asarray(nii.dataobj, dtype=np.float32)
    brain = np.asarray(bn.dataobj) > 0
    lesion = np.asarray(ln.dataobj) > 0
    if not np.isfinite(raw).all():
        raise ValueError("Nonfinite input")
    missing = (
        binary_dilation(lesion, iterations=a.dilation) if a.dilation else lesion
    ) & brain
    if not missing.any():
        raise ValueError("Empty inpaint mask")
    observed = brain & (distance_transform_edt(~lesion) > max(5, a.dilation))
    lo, hi = np.percentile(raw[observed], [1, 99])
    if hi <= lo:
        raise ValueError("Degenerate intensities")
    norm = np.clip(2 * (raw - lo) / (hi - lo) - 1, -1, 1).astype(np.float32)
    norm[~brain] = -1
    saved = torch.load(a.run / "best.pt", map_location="cpu", weights_only=True)
    if saved.get("kind") != "mixed_conditional_image_rf_3d_v1" or saved[
        "patch_size"
    ] != [96, 96, 96]:
        raise ValueError("Expected trained mixed RF96 checkpoint")
    if saved["identity"].get("functional_only"):
        raise ValueError("Preflight checkpoints are not trained models")
    device = torch.device(a.device)
    model = make_prior(saved["model_config"]).to(device).eval()
    model.load_state_dict(saved["model"])
    result = raw.copy()
    plans = plan_tiles(missing)
    coverage = np.zeros(raw.shape, np.uint8)
    for i, (start, owned) in enumerate(plans):
        m = extract_patch(missing, start, fill=False)
        context = np.where(m, 0, extract_patch(norm, start, fill=-1)).astype(np.float32)
        origin = nii.affine[:3, :3] @ start + nii.affine[:3, 3]
        pred = sample(
            model,
            torch.from_numpy(context[None, None]).to(device),
            torch.from_numpy(m[None, None].astype(np.float32)).to(device),
            torch.tensor(origin, dtype=torch.float32, device=device)[None],
            a.seed + i,
        )
        local = owned - start
        fill = pred[0, 0].cpu().numpy()[tuple(local.T)]
        result[tuple(owned.T)] = (fill + 1) / 2 * (hi - lo) + lo
        coverage[tuple(owned.T)] += 1
        print(f"tile={i + 1}/{len(plans)}", flush=True)
    if (
        not np.isfinite(result).all()
        or not np.array_equal(result[~missing], raw[~missing])
        or not np.all(coverage[missing] == 1)
    ):
        raise RuntimeError("Invalid output or coverage")
    a.output.parent.mkdir(parents=True, exist_ok=True)
    header = nii.header.copy()
    header.set_data_dtype(np.float32)
    nib.save(nib.Nifti1Image(result, nii.affine, header), a.output)
    nib.save(
        nib.Nifti1Image(missing.astype(np.uint8), nii.affine),
        str(a.output) + ".mask.nii.gz",
    )
    atomic_json(
        {
            "step": saved["step"],
            "tiles": len(plans),
            "outside_mask_max_error": 0,
            "normalization_percentiles": [float(lo), float(hi)],
            "seed": a.seed,
            "dilation": a.dilation,
        },
        str(a.output) + ".json",
    )


if __name__ == "__main__":
    main()
