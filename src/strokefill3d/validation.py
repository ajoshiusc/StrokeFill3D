"""Fixed held-out synthetic reconstruction evaluation, never training-loss selection."""

from __future__ import annotations

import csv
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from monai.metrics import SSIMMetric
from monai.metrics.regression import compute_ssim_and_cs

from .anatomy_data import AnatomicalT1Dataset
from .data import HealthyT1Dataset


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_split(
    train_manifest, validation_manifest, mask_manifest=None, test_manifest=None
):
    def rows(p):
        with open(p, newline="") as f:
            return list(csv.DictReader(f))

    train, val = rows(train_manifest), rows(validation_manifest)
    if not train or not val:
        raise ValueError("Training and validation manifests must be nonempty")
    groups = [train, val]
    if test_manifest:
        groups.append(rows(test_manifest))
    if test_manifest and not groups[-1]:
        raise ValueError("Test manifest is empty")
    for i, a in enumerate(groups):
        for b in groups[i + 1 :]:
            subjects_a = {str(Path(r["healthy_t1"]).resolve().parent.parent) for r in a}
            subjects_b = {str(Path(r["healthy_t1"]).resolve().parent.parent) for r in b}
            if subjects_a & subjects_b:
                raise ValueError("Subject overlap across train/validation/test")
            sites_a = {r["site"] for r in a if r.get("site")}
            sites_b = {r["site"] for r in b if r.get("site")}
            if sites_a & sites_b:
                raise ValueError("Site overlap across train/validation/test")
    if mask_manifest:
        allowed = {
            str(Path(r["exclude_mask"]).resolve())
            for r in train
            if r.get("exclude_mask")
        }
        donors = {str(Path(r["lesion_mask"]).resolve()) for r in rows(mask_manifest)}
        if not donors.issubset(allowed):
            raise ValueError("Donor bank must contain only training-scan masks")


def dataset_from_config(cfg, manifest, mask_manifest):
    common = {
        "patch_size": tuple(cfg["patch_size"]),
        "samples_per_scan": cfg["samples_per_scan"],
        "spacing": tuple(cfg["spacing_mm"]),
        "mask_manifest": mask_manifest,
        "max_mask_dilation_voxels": cfg.get("max_mask_dilation_voxels", 0),
    }
    if cfg.get("anatomy_channels", 0):
        return AnatomicalT1Dataset(
            manifest,
            **common,
            partial_mask_probability=cfg.get("partial_mask_probability", 0.35),
            large_mask_probability=cfg.get("large_mask_probability", 0.6),
            exclusion_margin=cfg.get("exclusion_margin", 3),
        )
    return HealthyT1Dataset(
        manifest,
        **common,
        cortical_patch_probability=cfg.get("cortical_patch_probability", 0),
        cortical_placement_probability=cfg.get("cortical_placement_probability", 0),
        cortical_surface_band_voxels=cfg.get("cortical_surface_band_voxels", 6),
        cortical_min_foreground_overlap=cfg.get(
            "cortical_min_foreground_overlap", 0.55
        ),
    )


def fixed_cases(cfg, manifest, donors, out):
    identity = {
        "validation_manifest": file_hash(manifest),
        "donors": file_hash(donors) if donors else None,
        "config": {
            k: cfg.get(k)
            for k in [
                "patch_size",
                "spacing_mm",
                "anatomy_channels",
                "validation_cases",
                "validation_seed",
                "exclusion_margin",
                "max_mask_dilation_voxels",
            ]
        },
        "schema": 2,
        "conditioning_code": file_hash(Path(__file__).with_name("anatomy.py")),
        "sampler_code": file_hash(Path(__file__).with_name("anatomy_data.py")),
    }
    path = Path(out) / "validation_cases.pt"
    if path.exists():
        saved = torch.load(path, map_location="cpu", weights_only=True)
        if saved["identity"] != identity:
            raise ValueError("Validation cache differs; use a fresh output directory")
        return saved["cases"]
    ds = dataset_from_config(cfg, manifest, donors)
    cases = []
    py_state = random.getstate()
    np_state = np.random.get_state()
    kinds = ["large_cortical", "large_deep", "partial_large_cortical", "small_deep"]
    try:
        with torch.random.fork_rng(devices=[]):
            for i in range(cfg.get("validation_cases", 8)):
                seed = cfg.get("validation_seed", 1729) + i
                random.seed(seed)
                np.random.seed(seed)
                if isinstance(ds, AnatomicalT1Dataset):
                    ds.requested_kind = kinds[i % len(kinds)]
                # Spread selected starting subjects through the held-out manifest.
                case = ds[
                    (i * max(1, len(ds.paths) // cfg.get("validation_cases", 8)))
                    % len(ds.paths)
                ]
                cases.append(case)
                print(
                    f"validation_case={i + 1} kind={case.get('kind', 'legacy')} mask_voxels={int(case['mask'].sum())}",
                    flush=True,
                )
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
    temporary = path.with_name(path.name + ".tmp")
    torch.save({"identity": identity, "cases": cases}, temporary)
    temporary.replace(path)
    (Path(out) / "validation_cases.json").write_text(
        json.dumps(
            [
                {
                    "index": i,
                    "source": c.get("source", ""),
                    "kind": c.get("kind", "legacy"),
                    "mask_voxels": int(c["mask"].sum()),
                }
                for i, c in enumerate(cases)
            ],
            indent=2,
        )
    )
    return cases


def masked_mean(value, mask):
    return (value * mask).sum() / mask.sum().clamp_min(1)


def interior_detail_metrics(pred, target, mask, ssim_map=None):
    """Score missing tissue away from copied context, in normalized [-1, 1] units.

    Gradient energy alone rewards noise. Report its ratio alongside alignment
    with the target gradients and error; none is a standalone realism score.
    A three-voxel erosion keeps each 7-voxel SSIM window inside the hole.
    Undefined scores (thin masks or constant targets) are explicitly null.
    """
    pred, target, mask = pred.float(), target.float(), mask.float()
    dims = target.ndim - 2
    if dims not in (2, 3) or pred.shape != target.shape or mask.shape != target.shape:
        raise ValueError("Expected aligned 2D or 3D prediction, target and mask")
    pool = F.max_pool2d if dims == 2 else F.max_pool3d
    interior = 1 - pool(F.pad(1 - mask, [3, 3] * dims, value=1), 7, 1)
    count = int(interior.sum())
    result = {
        "interior_voxels": count,
        "interior_ssim": None,
        "interior_gradient_cosine": None,
        "interior_gradient_ratio": None,
        "interior_gradient_mae": None,
    }
    if not count:
        return result
    if ssim_map is None:
        ssim_map, _ = compute_ssim_and_cs(
            F.pad(pred, [3, 3] * dims, mode="reflect"),
            F.pad(target, [3, 3] * dims, mode="reflect"),
            dims,
            (7,) * dims,
            (1.5,) * dims,
            data_range=2.0,
        )
    result["interior_ssim"] = float(masked_mean(ssim_map, interior))
    dot = pred_energy = target_energy = pred_abs = target_abs = error = pairs = 0
    for axis in range(2, target.ndim):
        n = target.shape[axis] - 1
        valid = interior.narrow(axis, 0, n) * interior.narrow(axis, 1, n)
        gp, gt = pred.diff(dim=axis), target.diff(dim=axis)
        dot = dot + (gp * gt * valid).sum()
        pred_energy = pred_energy + (gp.square() * valid).sum()
        target_energy = target_energy + (gt.square() * valid).sum()
        pred_abs = pred_abs + (gp.abs() * valid).sum()
        target_abs = target_abs + (gt.abs() * valid).sum()
        error = error + ((gp - gt).abs() * valid).sum()
        pairs = pairs + valid.sum()
    if pairs > 0:
        result["interior_gradient_mae"] = float(error / pairs)
    if target_abs > 1e-8:
        result["interior_gradient_ratio"] = float(pred_abs / target_abs)
        result["interior_gradient_cosine"] = float(
            dot / (pred_energy * target_energy).sqrt().clamp_min(1e-8)
        )
    return result


def reconstruction_metrics(pred, target, mask):
    """Metrics on known targets. WM proxy uses target intensity, not a tissue label."""
    pred = pred.float()
    target = target.float()
    mask = mask.float()
    error = (pred - target).abs()
    mse = masked_mean(error.square(), mask)
    dims = target.ndim - 2
    metric = SSIMMetric(spatial_dims=dims, data_range=2.0, reduction="none", win_size=7)
    # Restrict evaluation to the mask bounding box; SSIM includes its immediate context.
    loc = torch.nonzero(mask[0, 0] > 0)
    lo = loc.min(0).values
    hi = loc.max(0).values + 1
    # Pad very thin boxes so the 7-voxel SSIM window remains defined.
    lo = (lo - 3).clamp_min(0)
    hi = torch.minimum(hi + 3, torch.as_tensor(mask.shape[2:], device=hi.device))
    # Keep even thin face-touching masks valid for the diagnostic bbox SSIM.
    for axis, length in enumerate(target.shape[2:]):
        if int(hi[axis] - lo[axis]) < 7:
            lo[axis] = max(0, min(int(lo[axis]), length - 7))
            hi[axis] = lo[axis] + 7
    sl = (slice(None), slice(None), *[slice(int(a), int(b)) for a, b in zip(lo, hi)])
    ssim = float(metric(pred[sl], target[sl]).mean())
    # Score SSIM at missing-voxel centers, including patch-face masks. A bbox
    # average can be dominated by the perfectly copied observed context.
    padded_pred = F.pad(pred, [3, 3] * dims, mode="reflect")
    padded_target = F.pad(target, [3, 3] * dims, mode="reflect")
    ssim_map, _ = compute_ssim_and_cs(
        padded_pred, padded_target, dims, (7,) * dims, (1.5,) * dims, data_range=2.0
    )
    masked_ssim = float(masked_mean(ssim_map, mask))
    gradients = []
    for axis in range(2, target.ndim):
        a = [slice(None)] * target.ndim
        b = a.copy()
        a[axis] = slice(1, None)
        b[axis] = slice(None, -1)
        aa, bb = tuple(a), tuple(b)
        valid = mask[aa] * mask[bb]
        gradients.append(
            masked_mean(
                ((pred[aa] - pred[bb]) - (target[aa] - target[bb])).abs(), valid
            )
        )
    gradient = torch.stack(gradients).mean()
    fg = target > -0.95
    threshold = torch.quantile(target[fg], 0.65) if fg.any() else target.new_tensor(1.0)
    wm = (target > threshold).float()
    max_pool = F.max_pool2d if dims == 2 else F.max_pool3d
    avg_pool = F.avg_pool2d if dims == 2 else F.avg_pool3d
    wm = (1 - max_pool(1 - wm, 3, 1, 1)) * mask
    high_p = pred - avg_pool(pred, 3, 1, 1)
    high_t = target - avg_pool(target, 3, 1, 1)
    wm_error = float(masked_mean((high_p - high_t).abs(), wm)) if wm.sum() else None
    return {
        "masked_mae": float(masked_mean(error, mask)),
        "masked_psnr": float(10 * torch.log10(4 / mse.clamp_min(1e-12))),
        "bbox_ssim": ssim,
        "masked_ssim": masked_ssim,
        "gradient_mae": float(gradient),
        "wm_proxy_highpass_mae": wm_error,
        "wm_proxy_voxels": int(wm.sum()),
        "outside_max_error": float(error[mask == 0].max())
        if (mask == 0).any()
        else 0.0,
        **interior_detail_metrics(pred, target, mask, ssim_map),
    }


@torch.inference_mode()
def evaluate(model, cases, cfg, device, out, step):
    out = Path(out) / "validation" / f"step_{step:08d}"
    out.mkdir(parents=True, exist_ok=True)
    was_training = model.training
    model.eval()
    records = []
    previews = {}
    try:
        for index, case in enumerate(cases):
            context = case["image"][None].to(device)
            target = case["target"][None].to(device)
            mask = case["mask"][None].to(device)
            anatomy = case.get("anatomy")
            anatomy = anatomy[None].to(device) if anatomy is not None else None
            for seed_index in range(cfg.get("validation_seeds", 2)):
                gen = torch.Generator(device="cpu").manual_seed(
                    cfg.get("validation_seed", 1729) + index * 100 + seed_index
                )
                noise = torch.randn(target.shape, generator=gen).to(device)
                sample = torch.where(mask > 0, noise, context)
                steps = cfg.get("validation_steps", 48)
                for j in range(steps):
                    times = torch.full(
                        (1,), j / steps * cfg["time_embedding_scale"], device=device
                    )
                    with torch.autocast(
                        device_type=device.type,
                        dtype=torch.float16,
                        enabled=device.type == "cuda" and cfg.get("amp", False),
                    ):
                        velocity = model(sample, context, mask, times, anatomy)
                    sample = torch.where(
                        mask > 0, sample + velocity.float() / steps, context
                    )
                sample = sample.clamp(-1, 1) * mask + target * (1 - mask)
                if not torch.isfinite(sample).all():
                    raise RuntimeError("Nonfinite validation reconstruction")
                row = reconstruction_metrics(sample, target, mask)
                row.update(
                    case=index,
                    seed=seed_index,
                    kind=case.get("kind", "legacy"),
                    source=case.get("source", ""),
                )
                records.append(row)
                print(
                    f"validation step={step} case={index + 1}/{len(cases)} seed={seed_index} mae={row['masked_mae']:.5f} masked_ssim={row['masked_ssim']:.5f}",
                    flush=True,
                )
                if seed_index == 0:
                    previews[f"case_{index}_prediction"] = sample.cpu()
                    previews[f"case_{index}_target"] = target.cpu()
                    previews[f"case_{index}_mask"] = mask.cpu()
                    _save_preview(
                        out / f"case_{index:02d}.png", context, target, sample, mask
                    )
        metrics = {
            k: float(np.mean([r[k] for r in records if r[k] is not None]))
            for k in [
                "masked_mae",
                "masked_psnr",
                "bbox_ssim",
                "masked_ssim",
                "gradient_mae",
                "outside_max_error",
            ]
        }
        # Explicit combined score: lower MAE/gradient error and higher structural similarity.
        metrics["selection_score"] = (
            metrics["masked_mae"]
            + 0.25 * (1 - metrics["masked_ssim"])
            + 0.25 * metrics["gradient_mae"]
        )
        metrics["step"] = step
        (out / "metrics.json").write_text(
            json.dumps(
                {"summary": metrics, "cases": records}, indent=2, allow_nan=False
            )
        )
        torch.save(previews, out / "reconstructions.pt")
        return metrics
    finally:
        model.train(was_training)


def _save_preview(path, context, target, pred, mask):
    # Standard-library grayscale PNG writer: no plotting dependency on the remote host.
    import struct
    import zlib

    arrays = [x[0, 0].cpu().numpy() for x in [context, target, pred]]
    m = mask[0, 0].cpu().numpy()
    center = np.round(np.argwhere(m).mean(0)).astype(int)
    rows = []
    for axis in range(3):
        panels = []
        for a in arrays:
            sl = np.take(a, center[axis], axis=axis).T
            panels.append(np.uint8(np.clip((sl + 1) * 127.5, 0, 255)))
        rows.append(np.concatenate(panels, axis=1))
    width = max(r.shape[1] for r in rows)
    image = np.concatenate(
        [np.pad(r, ((0, 0), (0, width - r.shape[1]))) for r in rows], axis=0
    )
    height, width = image.shape

    def chunk(kind, data):
        return (
            struct.pack("!I", len(data))
            + kind
            + data
            + struct.pack("!I", zlib.crc32(kind + data) & 0xFFFFFFFF)
        )

    data = b"".join(b"\x00" + row.tobytes() for row in image)
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack("!2I5B", width, height, 8, 0, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(data))
        + chunk(b"IEND", b"")
    )
