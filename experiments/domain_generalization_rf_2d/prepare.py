"""Prepare lesion-free mixed-domain 2D RF targets and synthetic holdouts.

Stroke/resection masks are used only for a conservative exclusion zone and as
binary contour templates. Lesion-case MR intensities are never targets.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from scipy import ndimage

GRID = (197, 233, 189)
AFFINE = np.array(
    [[1, 0, 0, -98], [0, 1, 0, -134], [0, 0, 1, -72], [0, 0, 0, 1]], dtype=float
)


def load_array(path, dtype=np.float32):
    nii = nib.load(str(path))
    if tuple(nii.shape) != GRID or not np.allclose(nii.affine, AFFINE, atol=1e-3):
        raise ValueError(f"Not canonical 1-mm MNI space: {path}")
    return np.asanyarray(nii.dataobj, dtype=dtype)


def extract_templates(mask_volume, min_area=120, max_area=1800):
    templates = []
    for y in range(54, 191):
        plane = mask_volume[:, y, :] > 0
        labels, n = ndimage.label(plane, structure=np.ones((3, 3), np.uint8))
        for label in range(1, n + 1):
            pts = np.argwhere(labels == label)
            if not min_area <= len(pts) <= max_area:
                continue
            lo, hi = pts.min(0), pts.max(0) + 1
            if max(hi - lo) > 60:
                continue
            crop = labels[lo[0] : hi[0], lo[1] : hi[1]] == label
            canvas = np.zeros((64, 64), np.uint8)
            x0, z0 = (64 - crop.shape[0]) // 2, (64 - crop.shape[1]) // 2
            canvas[x0 : x0 + crop.shape[0], z0 : z0 + crop.shape[1]] = crop
            templates.append(canvas)
    return templates


def normalize_t1(image, brain):
    vals = image[brain & np.isfinite(image)]
    if len(vals) < 10000:
        raise ValueError("Brain mask is empty or too small")
    lo, hi = np.percentile(vals, [1, 99])
    if hi <= lo:
        raise ValueError("Invalid intensity range")
    out = np.clip((image - lo) / (hi - lo) * 2 - 1, -1, 1).astype(np.float32)
    out[~brain] = -1
    return out


def translated(template, cx, cz):
    sx, sz = int(cx) - 32, int(cz) - 32
    out = np.zeros((64, 64), np.uint8)
    x0, x1 = max(0, sx), min(64, 64 + sx)
    z0, z1 = max(0, sz), min(64, 64 + sz)
    if x1 > x0 and z1 > z0:
        out[x0:x1, z0:z1] = template[x0 - sx : x1 - sx, z0 - sz : z1 - sz]
    return out


def select_center(target, template, rng):
    points = []
    for cx in range(20, 45, 2):
        for cz in range(18, 47, 2):
            m = translated(template, cx, cz).astype(bool)
            n = int(m.sum())
            if n < 100:
                continue
            frac = ((target > -0.95) & m).sum() / n
            wm = ((target > 0.5) & m).sum() / n
            gm = (((target > -0.5) & (target < 0.4)) & m).sum() / n
            if frac >= 0.82 and wm >= 0.04 and gm >= 0.04:
                points.append((cx, cz))
    if not points:
        return None
    return points[int(rng.integers(len(points)))]


def sample_clean_patches(image, brain, exclusion, templates, domain, n, rng):
    targets, starts, centers, masks, groups = [], [], [], [], []
    if len(templates) < 10:
        raise RuntimeError(f"Too few lesion contours for {domain}: {len(templates)}")
    attempts = 0
    while len(targets) < n and attempts < n * 600:
        attempts += 1
        x = int(rng.integers(0, 134))
        z = int(rng.integers(38, 126))
        y = int(rng.integers(54, 191))
        sl = (slice(x, x + 64), y, slice(z, z + 64))
        if exclusion[sl].any():
            continue
        target = image[sl].copy()
        if brain[sl].mean() < 0.16 or float(target.std()) < 0.07:
            continue
        template = templates[int(rng.integers(len(templates)))]
        center = select_center(target, template, rng)
        if center is None:
            continue
        targets.append(target[None].astype(np.float16))
        starts.append([x, y - 134, z])
        centers.append(center)
        masks.append(translated(template, *center)[None].astype(np.uint8))
        groups.append(domain)
    if len(targets) < n:
        raise RuntimeError(
            f"Only found {len(targets)}/{n} complete target patches for {domain}"
        )
    return targets, starts, centers, masks, groups


def load_source(t1_path, brain_path, lesion_path, domain, count, rng):
    image = load_array(t1_path)
    brain = load_array(brain_path) > 0
    lesion = load_array(lesion_path) > 0
    # Use a 5-mm 3D margin so adjacent lesion effects do not become targets.
    exclusion = ndimage.distance_transform_edt(~lesion) <= 5.0
    image = normalize_t1(image, brain)
    templates = extract_templates(lesion)
    if not templates:
        raise RuntimeError(f"No lesion-mask contours found in {lesion_path}")
    return (
        *sample_clean_patches(image, brain, exclusion, templates, domain, count, rng),
        templates,
    )


def append_rows(dst, rows):
    for key, values in zip(["target", "starts", "centers", "mask", "domain"], rows):
        dst[key].extend(values)


def tensorize(d):
    return {
        "target": torch.from_numpy(np.stack(d["target"])),
        "starts": torch.tensor(d["starts"], dtype=torch.float32),
        "centers": torch.tensor(d["centers"], dtype=torch.float32),
        "mask": torch.from_numpy(np.stack(d["mask"])).bool(),
        "domain": d["domain"],
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episurg-medium", type=Path, required=True)
    p.add_argument("--epibios-medium", type=Path, required=True)
    p.add_argument("--episurg-heldout", type=Path, required=True)
    p.add_argument("--epibios-heldout", type=Path, required=True)
    p.add_argument("--arc-manifest", type=Path, required=True)
    p.add_argument("--arc-train-subjects", nargs="+", required=True)
    p.add_argument("--arc-heldout-subjects", nargs="+", required=True)
    p.add_argument("--train-patches-per-volume", type=int, default=256)
    p.add_argument("--epi-heldout-patches", type=int, default=32)
    p.add_argument("--arc-heldout-patches-per-session", type=int, default=12)
    p.add_argument("--seed", type=int, default=271128)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    rng = np.random.default_rng(args.seed)
    train = {k: [] for k in ["target", "starts", "centers", "mask", "domain"]}
    test = {k: [] for k in ["target", "starts", "centers", "mask", "domain"]}
    train_templates, heldout_templates, volumes = [], [], []

    def add_epi(case_dir, domain, split, count):
        case_dir = Path(case_dir)
        rows = load_source(
            case_dir / "t1_brain_training.nii.gz",
            case_dir / "brain_mask_training.nii.gz",
            case_dir / "lesion_training.nii.gz",
            domain,
            count,
            rng,
        )
        arrays, templates = rows[:-1], rows[-1]
        append_rows(train if split == "train" else test, arrays)
        (train_templates if split == "train" else heldout_templates).extend(templates)
        volumes.append(
            {
                "domain": domain,
                "split": split,
                "contours": len(templates),
                "patches": count,
            }
        )

    add_epi(args.episurg_medium, "episurg", "train", args.train_patches_per_volume)
    add_epi(args.epibios_medium, "epibios", "train", args.train_patches_per_volume)
    add_epi(args.episurg_heldout, "episurg", "test", args.epi_heldout_patches)
    add_epi(args.epibios_heldout, "epibios", "test", args.epi_heldout_patches)

    arc_rows = list(csv.DictReader(args.arc_manifest.open(newline="")))
    train_rows = [r for r in arc_rows if r["subject"] in args.arc_train_subjects]
    test_rows = [r for r in arc_rows if r["subject"] in args.arc_heldout_subjects]
    if set(args.arc_train_subjects) & set(args.arc_heldout_subjects):
        raise ValueError("ARC participants cannot occur in both splits")
    if not train_rows or not test_rows:
        raise RuntimeError("ARC split did not select both training and held-out scans")
    for split, source_rows, bucket, donor_bucket in [
        ("train", train_rows, train, train_templates),
        ("test", test_rows, test, heldout_templates),
    ]:
        counts = {}
        for r in source_rows:
            image = load_array(r["brain_t1_mni"])
            brain = load_array(r["skullstrip_mask_mni"]) > 0
            lesion = load_array(r["lesion_mask_mni"]) > 0
            exclusion = ndimage.distance_transform_edt(~lesion) <= 5.0
            image = normalize_t1(image, brain)
            templates = extract_templates(lesion)
            if len(templates) < 5:
                continue
            num = (
                args.train_patches_per_volume
                if split == "train"
                else args.arc_heldout_patches_per_session
            )
            arrays = sample_clean_patches(
                image, brain, exclusion, templates, "arc", num, rng
            )
            append_rows(bucket, arrays)
            donor_bucket.extend(templates)
            counts[r["subject"]] = counts.get(r["subject"], 0) + num
        volumes.append(
            {"domain": "arc", "split": split, "participant_sessions": counts}
        )

    if len(train_templates) < 30 or len(heldout_templates) < 30:
        raise RuntimeError("Insufficient real contour geometry")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "train": tensorize(train),
            "test": tensorize(test),
            "train_mask_templates": torch.from_numpy(
                np.stack(train_templates)[:, None]
            ).bool(),
            "heldout_mask_templates": torch.from_numpy(
                np.stack(heldout_templates)[:, None]
            ).bool(),
            "metadata": {
                "train_arc_subjects": args.arc_train_subjects,
                "heldout_arc_subjects": args.arc_heldout_subjects,
                "lesion_exclusion_mm": 5,
                "targets_are_complete_clean_patches": True,
                "lesion_image_intensities_used_as_targets": False,
                "volumes": volumes,
                "patch_counts": {
                    "train": len(train["target"]),
                    "test": len(test["target"]),
                },
            },
        },
        args.output,
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "volumes": volumes,
                "patch_counts": {
                    "train": len(train["target"]),
                    "test": len(test["target"]),
                },
                "mask_templates": {
                    "train": len(train_templates),
                    "heldout": len(heldout_templates),
                },
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
