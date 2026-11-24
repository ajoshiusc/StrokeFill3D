"""MONAI image loading and patch sampling for masked flow inpainting."""
from __future__ import annotations

import csv
import random
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import distance_transform_edt, gaussian_filter, zoom
from monai.transforms import Compose, LoadImage, Orientation, ScaleIntensityRangePercentiles, Spacing
from torch.utils.data import Dataset


class HealthyT1Dataset(Dataset):
    """Make known-target pairs from healthy or nonlesional T1 patches.

    The training CSV uses `healthy_t1` and may include an `exclude_mask` column
    containing the source scan's lesion mask. If present, patches intersecting
    that mask or a margin around it are rejected. A separate mask manifest with
    `lesion_mask` paths supplies ATLAS3 donor lesion shapes.
    """
    def __init__(self, manifest: str | Path, patch_size=(64, 64, 64), samples_per_scan=4,
                 spacing=(1.0, 1.0, 1.0), mask_manifest: str | Path | None = None):
        with open(manifest, newline="", encoding="utf8") as f:
            rows = list(csv.DictReader(f))
        if not rows or "healthy_t1" not in rows[0]:
            raise ValueError("Manifest must contain a non-empty healthy_t1 column")
        self.paths = [Path(r["healthy_t1"]).expanduser() for r in rows]
        self.exclude_paths = [Path(r["exclude_mask"]).expanduser() if r.get("exclude_mask") else None for r in rows]
        missing = [str(p) for p in self.paths if not p.is_file()]
        missing += [str(p) for p in self.exclude_paths if p is not None and not p.is_file()]
        if missing:
            raise FileNotFoundError(f"Missing {len(missing)} training files; first: {missing[0]}")
        self.patch_size = tuple(int(x) for x in patch_size)
        self.samples_per_scan = int(samples_per_scan)
        self.spacing = tuple(float(x) for x in spacing)
        self.pre = Compose([
            LoadImage(image_only=True, ensure_channel_first=True),
            Orientation(axcodes="RAS"),
            Spacing(pixdim=self.spacing, mode="bilinear"),
            ScaleIntensityRangePercentiles(lower=1, upper=99, b_min=-1, b_max=1, clip=True),
        ])
        self.mask_pre = Compose([
            LoadImage(image_only=True, ensure_channel_first=True),
            Orientation(axcodes="RAS"),
            Spacing(pixdim=self.spacing, mode="nearest"),
        ])
        # Bound host RAM: each loaded volume can occupy hundreds of MiB after resampling.
        self._cache: OrderedDict[int, tuple[torch.Tensor, torch.Tensor | None]] = OrderedDict()
        self.image_cache_size = 2
        self._mask_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self.mask_cache_size = 128
        self.mask_paths: list[Path] = []
        if mask_manifest:
            with open(mask_manifest, newline="", encoding="utf8") as f:
                mask_rows = list(csv.DictReader(f))
            if not mask_rows or "lesion_mask" not in mask_rows[0]:
                raise ValueError("Mask manifest must contain a non-empty lesion_mask column")
            self.mask_paths = [Path(r["lesion_mask"]).expanduser() for r in mask_rows]
            missing_masks = [str(p) for p in self.mask_paths if not p.is_file()]
            if missing_masks:
                raise FileNotFoundError(f"Missing {len(missing_masks)} donor masks; first: {missing_masks[0]}")

    def __len__(self):
        return len(self.paths) * self.samples_per_scan

    def _load(self, i: int) -> tuple[torch.Tensor, torch.Tensor | None]:
        if i in self._cache:
            self._cache.move_to_end(i)
            return self._cache[i]
        volume = self.pre(str(self.paths[i])).float()
        exclusion = None
        if self.exclude_paths[i] is not None:
            exclusion = (self.mask_pre(str(self.exclude_paths[i])) > 0).float()
            if volume.shape[1:] != exclusion.shape[1:] or not np.allclose(
                volume.affine.cpu().numpy(), exclusion.affine.cpu().numpy(), atol=1e-4
            ):
                raise ValueError(f"T1 and exclusion mask do not align after preprocessing: {self.paths[i]}")
        result = (volume, exclusion)
        self._cache[i] = result
        while len(self._cache) > self.image_cache_size:
            self._cache.popitem(last=False)
        return result

    def __getitem__(self, index):
        scan_index = index % len(self.paths)
        retryable = ("No lesion-free patch centers", "Could not sample a lesion-free patch",
                     "is smaller than patch")
        for _ in range(32):
            try:
                return self._sample_scan(scan_index)
            except ValueError as exc:
                if not any(message in str(exc) for message in retryable):
                    raise
                # Some scans cannot supply a 64³ clean target patch. Substitute
                # another scan rather than letting an ineligible item stop a worker.
                scan_index = random.randrange(len(self.paths))
        raise RuntimeError("Could not find a scan with a lesion-free target patch after 32 attempts")

    def _sample_scan(self, scan_index):
        vol, exclusion = self._load(scan_index)
        _, d, h, w = vol.shape
        pd, ph, pw = self.patch_size
        dims = (d, h, w)
        if any(dims[k] < self.patch_size[k] for k in range(3)):
            raise ValueError(f"Volume {self.paths[scan_index]} is smaller than patch {self.patch_size}")
        foreground = vol[0] > -0.95
        excluded_np = None
        if exclusion is not None:
            excluded_np = exclusion[0].cpu().numpy() > 0
            # Conservative Euclidean margin guarantees sampled cubic patches miss lesion voxels.
            radius = np.sqrt(sum((n // 2 + 2) ** 2 for n in self.patch_size))
            distances = distance_transform_edt(~excluded_np)
            safe = foreground.cpu().numpy() & (distances > radius)
            candidates = np.argwhere(safe)
            if len(candidates) == 0:
                raise ValueError(f"No lesion-free patch centers for {self.paths[scan_index]} at patch={self.patch_size}")
        else:
            candidates = torch.nonzero(foreground).cpu().numpy()
        target = None; starts = None
        for _ in range(32):
            if len(candidates) and random.random() < 0.9:
                center = candidates[random.randrange(len(candidates))].tolist()
            else:
                center = [random.randrange(d), random.randrange(h), random.randrange(w)]
            trial = [max(0, min(int(center[k]) - self.patch_size[k] // 2, dims[k] - self.patch_size[k])) for k in range(3)]
            if excluded_np is not None:
                ex_slice = tuple(slice(trial[k], trial[k] + self.patch_size[k]) for k in range(3))
                if excluded_np[ex_slice].any():
                    continue
            starts = trial
            target = vol[:, starts[0]:starts[0]+pd, starts[1]:starts[1]+ph, starts[2]:starts[2]+pw].clone()
            break
        if target is None:
            raise ValueError(f"Could not sample a lesion-free patch from {self.paths[scan_index]}")
        if self.mask_paths:
            donor = None
            for _ in range(min(32, len(self.mask_paths))):
                path = random.choice(self.mask_paths)
                try:
                    donor = self._load_donor_mask(path)
                    break
                except ValueError as exc:
                    if "Empty lesion mask" not in str(exc):
                        raise
                    self.mask_paths.remove(path)
            mask = (place_donor_mask(donor, self.patch_size, target[0] > -0.95)
                    if donor is not None else torch.zeros((1, *self.patch_size), dtype=torch.float32))
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
        low = coords.min(axis=0); high = coords.max(axis=0) + 1
        cropped = full_mask[tuple(slice(int(a), int(b)) for a, b in zip(low, high))]
        self._mask_cache[key] = cropped
        while len(self._mask_cache) > self.mask_cache_size:
            self._mask_cache.popitem(last=False)
        return cropped


def place_donor_mask(donor: np.ndarray, shape, foreground: torch.Tensor) -> torch.Tensor:
    """Flip/rotate and place a real lesion shape into a nonlesional target patch."""
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
    foreground_np = foreground.cpu().numpy()
    best = None; best_score = -1.0
    for _ in range(24):
        point = fg[random.randrange(len(fg))].tolist()
        starts = [max(0, min(int(point[k] - mask_np.shape[k] // 2), shape[k] - mask_np.shape[k])) for k in range(3)]
        canvas = np.zeros(shape, dtype=bool)
        sl = tuple(slice(starts[k], starts[k] + mask_np.shape[k]) for k in range(3))
        canvas[sl] = mask_np
        score = float((canvas & foreground_np).sum()) / max(int(canvas.sum()), 1)
        if score > best_score:
            best, best_score = canvas, score
        if score >= 0.65:
            break
    if best is None or best_score < 0.35:
        return torch.zeros((1, *shape), dtype=torch.float32)
    return torch.from_numpy(best[None].astype(np.float32))


def make_lesion_mask(shape, max_attempts=30) -> torch.Tensor:
    """Fallback connected irregular mask sampler for runs without ATLAS donors."""
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
