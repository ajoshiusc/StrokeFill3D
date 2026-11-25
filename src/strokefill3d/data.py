"""MONAI image loading and patch sampling for masked flow inpainting."""
from __future__ import annotations

import csv
import random
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import binary_dilation, distance_transform_edt, gaussian_filter, zoom
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
                 spacing=(1.0, 1.0, 1.0), mask_manifest: str | Path | None = None,
                 cortical_patch_probability=0.0, cortical_surface_band_voxels=6,
                 cortical_placement_probability=0.0, cortical_min_foreground_overlap=0.55,
                 max_mask_dilation_voxels=0):
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
        self.cortical_patch_probability = float(cortical_patch_probability)
        self.cortical_surface_band_voxels = int(cortical_surface_band_voxels)
        self.cortical_placement_probability = float(cortical_placement_probability)
        self.cortical_min_foreground_overlap = float(cortical_min_foreground_overlap)
        self.max_mask_dilation_voxels = int(max_mask_dilation_voxels)
        for name, value in (("cortical_patch_probability", self.cortical_patch_probability),
                            ("cortical_placement_probability", self.cortical_placement_probability)):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.cortical_surface_band_voxels < 1 or self.max_mask_dilation_voxels < 0:
            raise ValueError("Surface band must be positive and mask dilation nonnegative")
        if not 0.0 < self.cortical_min_foreground_overlap <= 1.0:
            raise ValueError("cortical_min_foreground_overlap must be in (0, 1]")
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
        self._surface_cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self.surface_cache_size = 2
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

    def _cortical_candidates(self, scan_index: int, foreground: np.ndarray) -> np.ndarray:
        """Return foreground voxels within a narrow band of the brain surface or sulcal CSF."""
        if scan_index in self._surface_cache:
            self._surface_cache.move_to_end(scan_index)
            return self._surface_cache[scan_index]
        inside_distance = distance_transform_edt(foreground)
        candidates = np.argwhere(foreground & (inside_distance <= self.cortical_surface_band_voxels))
        self._surface_cache[scan_index] = candidates
        while len(self._surface_cache) > self.surface_cache_size:
            self._surface_cache.popitem(last=False)
        return candidates

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
                     "Could not place a lesion mask in target foreground",
                     "mask lacks the required patch-edge margin", "is smaller than patch")
        for _ in range(32):
            try:
                return self._sample_scan(scan_index)
            except ValueError as exc:
                if not any(message in str(exc) for message in retryable):
                    raise
                # Some scans cannot supply a clean target patch at the configured size. Substitute
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
        foreground_np = foreground.cpu().numpy()
        safe_foreground = foreground_np
        excluded_np = None
        if exclusion is not None:
            excluded_np = exclusion[0].cpu().numpy() > 0
            # Reject target patches that touch the patient's actual lesion.
            radius = np.sqrt(sum((n // 2 + 2) ** 2 for n in self.patch_size))
            distances = distance_transform_edt(~excluded_np)
            safe_foreground = foreground_np & (distances > radius)
        candidates = np.argwhere(safe_foreground)
        if len(candidates) == 0:
            raise ValueError(f"No lesion-free patch centers for {self.paths[scan_index]} at patch={self.patch_size}")
        cortical_candidates = np.empty((0, 3), dtype=np.int64)
        if self.cortical_patch_probability or self.cortical_placement_probability:
            cortical_candidates = self._cortical_candidates(scan_index, foreground_np)
        if exclusion is not None and len(cortical_candidates):
            cortical_candidates = cortical_candidates[
                safe_foreground[tuple(cortical_candidates.T)]
            ]
        target = None; starts = None; centered_on_cortex = False
        for _ in range(32):
            centered_on_cortex = bool(len(cortical_candidates) and
                                      random.random() < self.cortical_patch_probability)
            if centered_on_cortex:
                center = cortical_candidates[random.randrange(len(cortical_candidates))].tolist()
            elif len(candidates) and random.random() < 0.9:
                center = candidates[random.randrange(len(candidates))].tolist()
            else:
                center = [random.randrange(d), random.randrange(h), random.randrange(w)]
            trial = [max(0, min(int(center[k]) - self.patch_size[k] // 2,
                                dims[k] - self.patch_size[k])) for k in range(3)]
            if excluded_np is not None:
                ex_slice = tuple(slice(trial[k], trial[k] + self.patch_size[k]) for k in range(3))
                if excluded_np[ex_slice].any():
                    continue
            starts = trial
            target = vol[:, starts[0]:starts[0]+pd, starts[1]:starts[1]+ph, starts[2]:starts[2]+pw].clone()
            break
        if target is None:
            raise ValueError(f"Could not sample a lesion-free patch from {self.paths[scan_index]}")
        target_foreground = target[0] > -0.95
        local_cortex = np.empty((0, 3), dtype=np.int64)
        if len(cortical_candidates):
            starts_np = np.asarray(starts)
            patch_end = starts_np + np.asarray(self.patch_size)
            inside = np.all((cortical_candidates >= starts_np) & (cortical_candidates < patch_end), axis=1)
            local_cortex = cortical_candidates[inside] - starts_np
        use_cortical_mask = bool(len(local_cortex) and
                                 random.random() < self.cortical_placement_probability)
        anchors = local_cortex if use_cortical_mask else None
        min_overlap = self.cortical_min_foreground_overlap if use_cortical_mask else 0.80
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
            mask = (place_donor_mask(donor, self.patch_size, target_foreground,
                                     anchor_candidates=anchors, min_foreground_overlap=min_overlap)
                    if donor is not None else torch.zeros((1, *self.patch_size), dtype=torch.float32))
            if mask.sum() < 20:
                mask = make_lesion_mask(target[0].shape, foreground=target_foreground)
        else:
            mask = make_lesion_mask(target[0].shape, foreground=target_foreground)
        if self.max_mask_dilation_voxels:
            iterations = random.randint(0, self.max_mask_dilation_voxels)
            if iterations:
                expanded = binary_dilation(mask[0].cpu().numpy() > 0, iterations=iterations)
                mask = torch.from_numpy(expanded[None].astype(np.float32))
        mask_voxels = mask[0] > 0
        overlap = float((mask_voxels & target_foreground).sum()) / max(int(mask_voxels.sum()), 1)
        if mask_voxels.sum() < 20 or overlap < min_overlap:
            raise ValueError(f"Could not place a lesion mask in target foreground (overlap={overlap:.3f})")
        mask_coords = torch.nonzero(mask_voxels)
        edge_margin = 4
        if bool((mask_coords < edge_margin).any()) or bool(
            (mask_coords >= torch.as_tensor(self.patch_size, device=mask_coords.device) - edge_margin).any()
        ):
            raise ValueError("Lesion mask lacks the required patch-edge margin")
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


def place_donor_mask(donor: np.ndarray, shape, foreground: torch.Tensor,
                     anchor_candidates: np.ndarray | None = None,
                     min_foreground_overlap: float = 0.80) -> torch.Tensor:
    """Transform and place a donor lesion with tissue overlap and edge context."""
    mask_np = donor.astype(bool, copy=True)
    for axis in range(3):
        if random.random() < 0.5:
            mask_np = np.flip(mask_np, axis=axis)
    mask_np = np.rot90(mask_np, k=random.randrange(4), axes=(1, 2)).copy()
    limit = tuple(max(8, int(n * 0.80)) for n in shape)
    factor = min(1.0, *(limit[k] / mask_np.shape[k] for k in range(3)))
    if factor < 1.0:
        mask_np = zoom(mask_np.astype(np.uint8), factor, order=0) > 0
    edge_margin = 4
    if not mask_np.any() or any(mask_np.shape[k] > shape[k] - 2 * edge_margin for k in range(3)):
        return torch.zeros((1, *shape), dtype=torch.float32)
    fg = torch.nonzero(foreground)
    if len(fg) == 0:
        return torch.zeros((1, *shape), dtype=torch.float32)
    foreground_np = foreground.cpu().numpy()
    anchors = np.asarray(anchor_candidates) if anchor_candidates is not None else np.empty((0, 3), dtype=np.int64)
    best = None; best_score = -1.0
    for _ in range(64):
        point = (anchors[random.randrange(len(anchors))].tolist() if len(anchors)
                 else fg[random.randrange(len(fg))].tolist())
        starts = [max(edge_margin, min(int(point[k] - mask_np.shape[k] // 2),
                                       shape[k] - mask_np.shape[k] - edge_margin)) for k in range(3)]
        canvas = np.zeros(shape, dtype=bool)
        sl = tuple(slice(starts[k], starts[k] + mask_np.shape[k]) for k in range(3))
        canvas[sl] = mask_np
        score = float((canvas & foreground_np).sum()) / max(int(canvas.sum()), 1)
        if score > best_score:
            best, best_score = canvas, score
        if score >= min(0.90, min_foreground_overlap + 0.10):
            break
    if best is None or best_score < min_foreground_overlap:
        return torch.zeros((1, *shape), dtype=torch.float32)
    return torch.from_numpy(best[None].astype(np.float32))


def make_lesion_mask(shape, max_attempts=120, foreground: torch.Tensor | np.ndarray | None = None,
                     min_foreground_overlap=0.80, edge_margin=4) -> torch.Tensor:
    """Sample an irregular lesion within foreground and away from patch borders."""
    d, h, w = map(int, shape)
    foreground_np = None
    candidates = None
    if foreground is not None:
        foreground_np = (foreground.detach().cpu().numpy() if torch.is_tensor(foreground)
                         else np.asarray(foreground)).astype(bool)
        candidates = np.argwhere(foreground_np)
        if len(candidates) == 0:
            return torch.zeros((1, d, h, w), dtype=torch.float32)
    for _ in range(max_attempts):
        center = (candidates[random.randrange(len(candidates))].tolist() if candidates is not None else
                  [random.randint(int(n * .2), max(int(n * .2), int(n * .8))) for n in (d, h, w)])
        radii = [random.uniform(max(2, n * .025), max(3, n * .16)) for n in (d, h, w)]
        zz, yy, xx = np.ogrid[:d, :h, :w]
        dist = ((zz-center[0])/radii[0])**2 + ((yy-center[1])/radii[1])**2 + ((xx-center[2])/radii[2])**2
        noise = gaussian_filter(np.random.randn(d, h, w).astype(np.float32), sigma=random.uniform(1.5, 4.0))
        threshold = np.quantile(noise, random.uniform(.58, .75))
        m = (dist <= 1.0) & (noise > threshold)
        if m.sum() >= 20:
            coords = np.argwhere(m)
            if np.any(coords.min(axis=0) < edge_margin) or np.any(
                coords.max(axis=0) >= np.asarray((d, h, w)) - edge_margin
            ):
                continue
            if foreground_np is not None:
                overlap = float((m & foreground_np).sum()) / max(int(m.sum()), 1)
                if overlap < min_foreground_overlap:
                    continue
            return torch.from_numpy(m[None].astype(np.float32))
    return torch.zeros((1, d, h, w), dtype=torch.float32)
