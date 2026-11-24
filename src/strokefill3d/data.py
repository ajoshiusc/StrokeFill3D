"""Manifest-backed healthy-T1 dataset with synthetic stroke-shaped masks."""
from __future__ import annotations

import csv
import random
from collections import OrderedDict
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from scipy.ndimage import gaussian_filter, zoom
from monai.transforms import Compose, LoadImage, Orientation, ScaleIntensityRangePercentiles, Spacing
from torch.utils.data import Dataset


class HealthyT1Dataset(Dataset):
    """Build input/target pairs from healthy T1 scans and sampled lesion masks.

    CSV requires a `healthy_t1` column. An optional second CSV can provide a
    `lesion_mask` column; donor masks are shape-transferred into target patches.
    Without a mask manifest, irregular synthetic masks are used.
    """
    def __init__(self, manifest: str | Path, patch_size=(48, 48, 48), samples_per_scan=24,
                 spacing=(1.0, 1.0, 1.0), mask_manifest: str | Path | None = None):
        with open(manifest, newline="", encoding="utf8") as f:
            rows = list(csv.DictReader(f))
        if not rows or "healthy_t1" not in rows[0]:
            raise ValueError("Manifest must contain a non-empty healthy_t1 column")
        self.paths = [Path(r["healthy_t1"]).expanduser() for r in rows]
        missing = [str(p) for p in self.paths if not p.is_file()]
        if missing:
            raise FileNotFoundError(f"Missing {len(missing)} T1 files; first: {missing[0]}")
        self.patch_size = tuple(int(x) for x in patch_size)
        self.samples_per_scan = int(samples_per_scan)
        self.pre = Compose([
            LoadImage(image_only=True, ensure_channel_first=True),
            Orientation(axcodes="RAS"),
            Spacing(pixdim=spacing, mode="bilinear"),
            ScaleIntensityRangePercentiles(lower=1, upper=99, b_min=-1, b_max=1, clip=True),
        ])
        self._cache: dict[int, torch.Tensor] = {}
        self.mask_paths: list[Path] = []
        self.mask_pre = Compose([
            LoadImage(image_only=True, ensure_channel_first=True),
            Orientation(axcodes="RAS"),
            Spacing(pixdim=spacing, mode="nearest"),
        ])
        self._mask_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self.mask_cache_size = 128
        if mask_manifest:
            with open(mask_manifest, newline="", encoding="utf8") as f:
                mask_rows = list(csv.DictReader(f))
            if not mask_rows or "lesion_mask" not in mask_rows[0]:
                raise ValueError("Mask manifest must contain a non-empty lesion_mask column")
            self.mask_paths = [Path(r["lesion_mask"]).expanduser() for r in mask_rows]
            missing_masks = [str(p) for p in self.mask_paths if not p.is_file()]
            if missing_masks:
                raise FileNotFoundError(f"Missing {len(missing_masks)} lesion masks; first: {missing_masks[0]}")

    def __len__(self):
        return len(self.paths) * self.samples_per_scan

    def _load(self, i: int) -> torch.Tensor:
        if i not in self._cache:
            tensor = self.pre(str(self.paths[i]))
            self._cache[i] = tensor.float()
        return self._cache[i]

    def __getitem__(self, index):
        vol = self._load(index % len(self.paths))
        _, d, h, w = vol.shape
        pd, ph, pw = self.patch_size
        if min(d, h, w) < min(pd, ph, pw):
            raise ValueError(f"Volume {self.paths[index % len(self.paths)]} is smaller than patch {self.patch_size}; preprocess to a common field of view")
        # Bias patch centers toward foreground, while retaining occasional context-only patches.
        foreground = torch.nonzero(vol[0] > -0.95)
        if len(foreground) and random.random() < 0.85:
            center = foreground[random.randrange(len(foreground))].tolist()
        else:
            center = [random.randrange(d), random.randrange(h), random.randrange(w)]
        starts = [max(0, min(center[k] - self.patch_size[k] // 2,
                             (d, h, w)[k] - self.patch_size[k])) for k in range(3)]
        target = vol[:, starts[0]:starts[0]+pd, starts[1]:starts[1]+ph, starts[2]:starts[2]+pw].clone()
        if self.mask_paths:
            donor = self._load_donor_mask(random.choice(self.mask_paths))
            mask = place_donor_mask(donor, self.patch_size, target[0] > -0.95)
            if mask.sum() < 20:
                mask = make_lesion_mask(target[0].shape)
        else:
            mask = make_lesion_mask(target[0].shape)
        masked = target.clone()
        masked[0][mask[0] > 0] = 0.0
        return {"image": masked, "mask": mask, "target": target}

    def _load_donor_mask(self, path: Path) -> np.ndarray:
        key = str(path)
        if key in self._mask_cache:
            self._mask_cache.move_to_end(key)
            return self._mask_cache[key]
        image = self.mask_pre(key)
        full_mask = np.asarray(image[0] > 0, dtype=np.uint8)
        coords = np.argwhere(full_mask)
        if coords.size == 0:
            raise ValueError(f"Empty lesion mask: {path}")
        low = coords.min(axis=0)
        high = coords.max(axis=0) + 1
        cropped = full_mask[tuple(slice(int(a), int(b)) for a, b in zip(low, high))]
        self._mask_cache[key] = cropped
        while len(self._mask_cache) > self.mask_cache_size:
            self._mask_cache.popitem(last=False)
        return cropped


def place_donor_mask(donor: np.ndarray, shape, foreground: torch.Tensor) -> torch.Tensor:
    """Randomly reorient and place a real lesion shape into a healthy patch."""
    mask_np = donor.astype(bool, copy=True)
    for axis in range(3):
        if random.random() < 0.5:
            mask_np = np.flip(mask_np, axis=axis)
    mask_np = np.rot90(mask_np, k=random.randrange(4), axes=(1, 2)).copy()
    limit = tuple(max(8, int(n * 0.88)) for n in shape)
    factor = min(1.0, *(limit[k] / mask_np.shape[k] for k in range(3)))
    if factor < 1.0:
        mask_np = zoom(mask_np.astype(np.uint8), factor, order=0) > 0
    if not mask_np.any() or any(mask_np.shape[k] > shape[k] for k in range(3)):
        return torch.zeros((1, *shape), dtype=torch.float32)
    fg = torch.nonzero(foreground)
    if len(fg) == 0:
        fg = torch.tensor([[n // 2 for n in shape]])
    best = None; best_score = -1.0
    for _ in range(24):
        point = fg[random.randrange(len(fg))].tolist()
        starts = [max(0, min(int(point[k] - mask_np.shape[k] // 2), shape[k] - mask_np.shape[k])) for k in range(3)]
        canvas = np.zeros(shape, dtype=bool)
        sl = tuple(slice(starts[k], starts[k] + mask_np.shape[k]) for k in range(3))
        canvas[sl] = mask_np
        count = int(canvas.sum())
        score = float((canvas & foreground.cpu().numpy()).sum()) / max(count, 1)
        if score > best_score:
            best, best_score = canvas, score
        if score >= 0.65:
            break
    if best is None or best_score < 0.35:
        return torch.zeros((1, *shape), dtype=torch.float32)
    return torch.from_numpy(best[None].astype(np.float32))


def make_lesion_mask(shape, max_attempts=30) -> torch.Tensor:
    """Sample connected, irregular ellipsoids spanning plausible lesion scales."""
    d, h, w = map(int, shape)
    for _ in range(max_attempts):
        center = [random.randint(int(n * .2), max(int(n * .2), int(n * .8))) for n in (d, h, w)]
        radii = [random.uniform(max(2, n * .025), max(3, n * .16)) for n in (d, h, w)]
        zz, yy, xx = np.ogrid[:d, :h, :w]
        dist = ((zz-center[0])/radii[0])**2 + ((yy-center[1])/radii[1])**2 + ((xx-center[2])/radii[2])**2
        noise = gaussian_filter(np.random.randn(d, h, w).astype(np.float32), sigma=random.uniform(1.5, 4.0))
        threshold = np.quantile(noise, random.uniform(.58, .75))
        m = (dist <= 1.0) & (noise > threshold)
        if m.sum() >= 20:
            return torch.from_numpy(m[None].astype(np.float32))
    return torch.zeros((1, d, h, w), dtype=torch.float32)
