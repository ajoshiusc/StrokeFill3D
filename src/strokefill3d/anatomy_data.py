"""Whole-volume mask placement followed by clean-target patch extraction."""

from __future__ import annotations

import random

import numpy as np
import torch
from scipy.ndimage import binary_dilation, distance_transform_edt, zoom

from .anatomy import anatomy_features, check_mni
from .data import HealthyT1Dataset


class AnatomicalT1Dataset(HealthyT1Dataset):
    """MNI cerebral sampling; regions are heuristics, not tissue segmentations."""

    def __init__(
        self,
        *args,
        partial_mask_probability=0.35,
        large_mask_probability=0.6,
        exclusion_margin=3,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.partial_probability = float(partial_mask_probability)
        self.large_probability = float(large_mask_probability)
        self.exclusion_margin = int(exclusion_margin)
        self.requested_kind = None
        self._region_cache = {}

    def _regions(self, i, a, affine, exclusion):
        if i in self._region_cache:
            return self._region_cache[i]
        brain = a > -0.95
        distance = distance_transform_edt(brain)
        z = np.arange(a.shape[2]) + affine[2, 3]
        # Explicitly restrict this profile's masks to supratentorial cerebral tissue.
        cerebral = brain & (z[None, None, :] > -10)
        safe = cerebral & ~exclusion
        cortical = np.argwhere(safe & (distance <= 6))
        deep = np.argwhere(safe & (distance >= 8))
        if not len(cortical) or not len(deep):
            raise ValueError("No lesion-free patch centers in cerebral regions")
        self._region_cache = {i: (cortical, deep)}
        return cortical, deep

    def _sample_scan(self, scan_index):
        vol, real_lesion = self._load(scan_index)
        a = np.asarray(vol[0], dtype=np.float32)
        affine = vol.affine.cpu().numpy()
        check_mni(affine, a.shape)
        shape = np.asarray(a.shape)
        size = np.asarray(self.patch_size)
        if (shape < size).any():
            raise ValueError("Volume is smaller than patch")
        excluded = (
            np.asarray(real_lesion[0]) > 0
            if real_lesion is not None
            else np.zeros(a.shape, dtype=bool)
        )
        if self.exclusion_margin:
            excluded = binary_dilation(excluded, iterations=self.exclusion_margin)
        cortical, deep = self._regions(scan_index, a, affine, excluded)
        kind = self.requested_kind
        is_cortical = ("cortical" in kind) if kind else random.random() < 0.6
        large = ("large" in kind) if kind else random.random() < self.large_probability
        partial = (
            ("partial" in kind) if kind else random.random() < self.partial_probability
        )
        points = cortical if is_cortical else deep
        for _ in range(96):
            anchor = points[random.randrange(len(points))]
            starts = np.clip(anchor - size // 2, 0, shape - size)
            if partial:
                axis = random.randrange(3)
                starts[axis] = np.clip(
                    anchor[axis] - random.choice([2, int(size[axis]) - 3]),
                    0,
                    shape[axis] - size[axis],
                )
            crop = tuple(slice(int(s), int(s + n)) for s, n in zip(starts, size))
            # Exact box rejection avoids the old overly conservative spherical radius.
            if excluded[crop].any():
                continue
            if self.mask_paths:
                donor = self._load_donor_mask(random.choice(self.mask_paths)).copy()
                if donor.sum() == 0:
                    continue
                for axis in range(3):
                    if random.random() < 0.5:
                        donor = np.flip(donor, axis)
                donor = np.rot90(donor, random.randrange(4), axes=(0, 1)).copy()
                desired = (
                    random.uniform(18000, 65000)
                    if large
                    else random.uniform(500, 10000)
                )
                factor = (desired / max(float(donor.sum()), 1)) ** (1 / 3)
                max_extent = float(size.min()) * (1.25 if partial else 0.80)
                factor = min(factor, max_extent / max(donor.shape))
                donor = zoom(donor, factor, order=0) > 0
            else:
                n = 64 if large else 24
                grid = np.ogrid[tuple(slice(-1, 1, complex(n)) for _ in range(3))]
                donor = sum(g * g for g in grid) < 1
            low = anchor - np.asarray(donor.shape) // 2
            high = low + np.asarray(donor.shape)
            dest_lo = np.maximum(low, 0)
            dest_hi = np.minimum(high, shape)
            if (dest_lo >= dest_hi).any():
                continue
            missing = np.zeros(a.shape, dtype=bool)
            dest = tuple(slice(int(l), int(h)) for l, h in zip(dest_lo, dest_hi))
            src = tuple(
                slice(int(l), int(h)) for l, h in zip(dest_lo - low, dest_hi - low)
            )
            missing[dest] = donor[src]
            dilation = random.randint(0, self.max_mask_dilation_voxels)
            if dilation:
                missing = binary_dilation(missing, iterations=dilation)
            local = missing[crop]
            if local.sum() < (12000 if large else 100):
                continue
            overlap = np.mean(a[crop][local] > -0.95)
            if overlap < (0.65 if is_cortical else 0.85):
                continue
            zz = np.arange(a.shape[2]) + affine[2, 3]
            if (
                np.mean(np.broadcast_to(zz[None, None, :] > -10, a.shape)[missing])
                < 0.9
            ):
                continue
            coords = np.argwhere(local)
            touches = bool(
                (coords.min(0) < 4).any() or (coords.max(0) >= size - 4).any()
            )
            if not partial and (touches or local.sum() != missing.sum()):
                continue
            if partial and not touches:
                continue
            target = torch.from_numpy(a[crop].copy()[None])
            mask = torch.from_numpy(local.copy()[None].astype(np.float32))
            context = target * (1 - mask)
            anatomy = anatomy_features(a, missing, affine, crop=crop, excluded=excluded)
            return {
                "image": context,
                "mask": mask,
                "target": target,
                "anatomy": anatomy,
                "source": str(self.paths[scan_index]),
                "kind": kind
                or f"{'partial_' if partial else ''}{'large' if large else 'small'}_{'cortical' if is_cortical else 'deep'}",
                "affine": torch.from_numpy(affine.copy()),
                "starts": torch.as_tensor(starts.copy()),
            }
        raise ValueError(
            "Could not sample a lesion-free patch with the requested mask stratum"
        )
