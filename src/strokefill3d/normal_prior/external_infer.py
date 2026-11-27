"""Inpaint preprocessed external MRI with adaptive 96-cube coverage.

Every fit hides the complete original injury mask within its context window.
Disjoint ownership cores avoid averaging generated anatomical structures.
"""

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from scipy.ndimage import binary_dilation

from .core import coordinate_maps, file_hash, fit_source, load_bundle
from .data import check_grid, normalize
from .io import RunLock, atomic_json


def plan_tiles(mask, size=96, margin=12):
    """Greedily cover masked voxels using interior ownership cores.

    Starts may be negative; patches are explicitly padded with background.
    Every masked voxel has at least ``margin`` model voxels to the patch edge.
    """
    if mask.ndim != 3 or not mask.any():
        raise ValueError("Require a nonempty 3D mask")
    if margin < 1 or 2 * margin >= size:
        raise ValueError("Invalid context margin")
    points = np.argwhere(mask)
    # Candidate centers follow the lesion, avoiding nearly empty regular tiles.
    centers = np.unique((points // 12) * 12 + 6, axis=0)
    centers = np.concatenate([centers, np.rint(points.mean(0))[None].astype(int)])
    starts = centers - size // 2
    low = np.full(3, -margin)
    high = np.asarray(mask.shape) - size + margin
    if (high < low).any():
        raise ValueError("Image is smaller than the supported ownership core")
    starts = np.unique(np.clip(starts, low, high), axis=0)
    remaining = mask.copy()
    plans = []
    while remaining.any():
        sums = (
            np.pad(remaining.astype(np.int32), ((1, 0),) * 3)
            .cumsum(0, dtype=np.int32)
            .cumsum(1, dtype=np.int32)
            .cumsum(2, dtype=np.int32)
        )
        lo = starts + margin
        hi = starts + size - margin
        a, b, c = lo.T
        x, y, z = hi.T
        counts = (
            sums[x, y, z]
            - sums[a, y, z]
            - sums[x, b, z]
            - sums[x, y, c]
            + sums[a, b, z]
            + sums[a, y, c]
            + sums[x, b, c]
            - sums[a, b, c]
        )
        best = int(counts.argmax())
        if counts[best] == 0:
            raise RuntimeError("Adaptive tile plan left injury voxels uncovered")
        start = starts[best]
        sl = tuple(slice(int(s + margin), int(s + size - margin)) for s in start)
        owned = np.argwhere(remaining[sl]) + start + margin
        remaining[tuple(owned.T)] = False
        plans.append((start.copy(), owned))
    return plans


def extract_patch(array, start, size=96, fill=0):
    shape = np.asarray(array.shape)
    lo = np.maximum(start, 0)
    hi = np.minimum(start + size, shape)
    patch = np.full((size,) * 3, fill, dtype=array.dtype)
    dst = tuple(slice(int(a), int(b)) for a, b in zip(lo - start, hi - start))
    src = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
    patch[dst] = array[src]
    return patch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--prepared", required=True, help="Completed normal_prior.preprocess directory"
    )
    p.add_argument("--run", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--checkpoint", default="best.pt")
    p.add_argument("--device", default="cuda")
    p.add_argument("--flow-steps", type=int, default=16)
    p.add_argument("--iterations", type=int, default=120)
    p.add_argument("--seed", type=int, default=909)
    p.add_argument("--mask-dilation-voxels", type=int, default=2)
    p.add_argument("--plan-only", action="store_true")
    args = p.parse_args()
    if args.mask_dilation_voxels < 0 or args.flow_steps < 1 or args.iterations < 0:
        raise ValueError("Invalid inference settings")
    prepared = Path(args.prepared)
    prep = json.loads((prepared / "preprocessing.json").read_text())
    qc = prep["qc"]
    if (
        qc["brain_mask_dice_with_template"] < 0.8
        or qc["nonpositive_jacobian_voxels_in_brain"]
    ):
        raise ValueError("Preprocessing registration QC failed")
    image_path = prepared / "t1_brain_training.nii.gz"
    lesion_path = prepared / "lesion_training.nii.gz"
    brain_path = prepared / "brain_mask_training.nii.gz"
    nii = nib.load(image_path)
    masknii = nib.load(lesion_path)
    bn = nib.load(brain_path)
    check_grid(nii, masknii)
    check_grid(nii, bn)
    raw = np.asarray(nii.dataobj, dtype=np.float32)
    brain = np.asarray(bn.dataobj) > 0.5
    original_mask = np.asarray(masknii.dataobj) > 0.5
    mask = original_mask.copy()
    if args.mask_dilation_voxels:
        mask = binary_dilation(mask, iterations=args.mask_dilation_voxels)
    # Brain extraction defines the domain. Save the effective mask explicitly.
    missing = mask & brain
    if not np.all(raw[~brain] == 0):
        raise ValueError("Prepared T1 has nonzero signal outside the brain mask")
    plans = plan_tiles(missing)
    print(
        json.dumps(
            {
                "tiles": len(plans),
                "effective_mask_voxels": int(missing.sum()),
                "original_lesion_voxels_outside_brain": int(
                    (original_mask & ~brain).sum()
                ),
                "owned_voxels_per_tile": [len(v) for _, v in plans],
            }
        ),
        flush=True,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.plan_only:
        return
    normalized, lo, hi = normalize(raw, missing)
    device = torch.device(args.device)
    model, codec, saved = load_bundle(args.run, args.checkpoint, args.device)
    if tuple(codec.patch_size) != (96, 96, 96):
        raise ValueError("External inference requires a trained 96-cube codec")
    identity = {
        "inputs": {str(v): file_hash(v) for v in [image_path, lesion_path, brain_path]},
        "codec_sha256": saved["codec_sha256"],
        "checkpoint_sha256": file_hash(
            Path(args.checkpoint)
            if Path(args.checkpoint).is_absolute()
            else Path(args.run) / "prior" / args.checkpoint
        ),
        "settings": vars(args),
    }
    cache = Path(str(output) + ".tiles")
    cache.mkdir(exist_ok=True)
    with RunLock(cache):
        if (cache / "identity.json").exists():
            if json.loads((cache / "identity.json").read_text()) != identity:
                raise ValueError(
                    "Tile cache belongs to different inputs/checkpoint/settings; use a new output path"
                )
        else:
            atomic_json(identity, cache / "identity.json")
        result = normalized.copy()
        coverage = np.zeros(raw.shape, dtype=np.uint8)
        records = []
        for i, (start, owned) in enumerate(plans):
            cached = cache / f"tile_{i:03d}.npz"
            historyfile = cache / f"tile_{i:03d}.json"
            if cached.exists() and historyfile.exists():
                values = np.load(cached)["values"]
                history = json.loads(historyfile.read_text())
            else:
                # Never use another generated tile as visible data, and never expose
                # any original labeled injury intensities to this fitting step.
                local_mask = extract_patch(missing, start, fill=False)
                context = extract_patch(normalized, start, fill=-1.0)
                context = np.where(local_mask, 0, context).astype(np.float32)
                image = torch.from_numpy(context[None, None]).to(device)
                mask_tensor = torch.from_numpy(
                    local_mask[None, None].astype(np.float32)
                ).to(device)
                origin = nii.affine[:3, :3] @ start + nii.affine[:3, 3]
                coords = coordinate_maps(
                    torch.tensor(origin, dtype=torch.float32, device=device)[None],
                    [v // codec.factor for v in codec.patch_size],
                    codec.factor,
                )
                pred, _, _, history = fit_source(
                    model,
                    codec,
                    image,
                    mask_tensor,
                    coords,
                    seed=args.seed + i,
                    steps=args.flow_steps,
                    iterations=args.iterations,
                    trust_weight=0.001,
                    boundary_weight=4.0,
                    checkpoint_steps=True,
                    callback=lambda row, i=i: print(
                        f"TILE {i + 1}/{len(plans)} " + json.dumps(row), flush=True
                    ),
                )
                local = owned - start
                values = pred[0, 0].cpu().numpy()[tuple(local.T)]
                np.savez_compressed(cached, values=values)
                atomic_json(history, historyfile)
            if values.shape != (len(owned),) or not np.isfinite(values).all():
                raise RuntimeError("Invalid tile output/cache")
            result[tuple(owned.T)] = values
            coverage[tuple(owned.T)] += 1
            records.append(
                {
                    "tile": i + 1,
                    "start": start.tolist(),
                    "owned_voxels": len(owned),
                    "history": history,
                }
            )
        if not np.all(coverage[missing] == 1) or coverage[~missing].any():
            raise RuntimeError("Invalid tile coverage")
        fill = (result + 1) * 0.5 * (hi - lo) + lo
        output_data = np.where(missing, fill, raw).astype(np.float32)
        if not np.isfinite(output_data).all() or not np.array_equal(
            output_data[~missing], raw[~missing]
        ):
            raise RuntimeError("Output changed observed voxels or is nonfinite")
        h = nii.header.copy()
        h.set_data_dtype(np.float32)
        nib.save(nib.Nifti1Image(output_data, nii.affine, h), output)
        h.set_data_dtype(np.uint8)
        nib.save(
            nib.Nifti1Image(missing.astype(np.uint8), nii.affine, h),
            str(output) + ".mask.nii.gz",
        )
        atomic_json(
            dict(
                **identity,
                checkpoint_step=saved["step"],
                tiles=records,
                normalization_percentiles=[lo, hi],
                normalization_uses_observed_voxels_only=True,
                outside_mask_max_error=0.0,
                mask_voxels=int(missing.sum()),
                original_lesion_voxels_outside_brain=int(
                    (original_mask & ~brain).sum()
                ),
                note="Experimental external-data inference; preprocessing and generated anatomy require visual review",
            ),
            str(output) + ".json",
        )
        print("DONE", output, flush=True)


if __name__ == "__main__":
    main()
