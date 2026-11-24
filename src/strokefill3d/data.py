"""Manifest-backed healthy-T1 dataset with synthetic stroke-shaped masks."""
from __future__ import annotations

import csv
import random
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from monai.transforms import Compose, LoadImage, Orientation, ScaleIntensityRangePercentiles, Spacing
from torch.utils.data import Dataset


class HealthyT1Dataset(Dataset):
    """Build input/target pairs from healthy T1 scans and sampled lesion masks.

    CSV requires a `healthy_t1` column with paths to authorized, defaced scans.
    The synthetic mask is generated in normalized voxel space at each epoch.
    """
    def __init__(self, manifest: str | Path, patch_size=(96, 96, 96), samples_per_scan=24,
                 spacing=(1.0, 1.0, 1.0)):
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
        foreground = torch.nonzero(vol[0].abs() > 0.05)
        if len(foreground) and random.random() < 0.85:
            center = foreground[random.randrange(len(foreground))].tolist()
        else:
            center = [random.randrange(d), random.randrange(h), random.randrange(w)]
        starts = [max(0, min(center[k] - self.patch_size[k] // 2,
                             (d, h, w)[k] - self.patch_size[k])) for k in range(3)]
        target = vol[:, starts[0]:starts[0]+pd, starts[1]:starts[1]+ph, starts[2]:starts[2]+pw].clone()
        mask = make_lesion_mask(target[0].shape)
        masked = target.clone()
        masked[0][mask[0] > 0] = 0.0
        return {"image": masked, "mask": mask, "target": target}


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
