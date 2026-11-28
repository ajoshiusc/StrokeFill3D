"""Stateless lesion-free 3D patch sampling and fixed cortical validation holes."""

import csv
from collections import OrderedDict
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from scipy.ndimage import (
    binary_dilation,
    binary_erosion,
    distance_transform_edt,
    uniform_filter,
    zoom,
)
from torch.utils.data import Dataset, Sampler


def normalize(raw, missing=None):
    values = raw if missing is None else raw[~missing]
    if not np.isfinite(values).all():
        raise ValueError("Observed MRI contains nonfinite values")
    lo, hi = np.percentile(values, [1, 99])
    if hi <= lo:
        raise ValueError("Degenerate MRI intensity range")
    return (
        np.clip((raw - lo) / (hi - lo) * 2 - 1, -1, 1).astype(np.float32),
        float(lo),
        float(hi),
    )


def check_grid(image, mask=None):
    if len(image.shape) != 3 or not np.allclose(
        image.affine[:3, :3], np.eye(3), atol=1e-3
    ):
        raise ValueError(
            "Expected an axis-aligned RAS 1-mm MNI image; registration is required, not inferred from its header"
        )
    if mask is not None and (
        mask.shape != image.shape
        or not np.allclose(mask.affine, image.affine, atol=1e-4)
    ):
        raise ValueError("T1 and lesion-mask grids differ")


class NormalPatches(Dataset):
    def __init__(
        self,
        manifest,
        patch_size=(96, 96, 96),
        seed=2026,
        exclusion_margin=3,
        cache_size=2,
    ):
        with open(manifest, newline="") as f:
            self.rows = list(csv.DictReader(f))
        if not self.rows or any(
            not r.get("healthy_t1") or not r.get("exclude_mask") for r in self.rows
        ):
            raise ValueError("Require healthy_t1 and exclude_mask columns")
        np.random.default_rng(seed).shuffle(self.rows)
        self.size = np.array(patch_size)
        self.seed = seed
        self.margin = exclusion_margin
        self.cache_size = cache_size
        self.cache = OrderedDict()
        for row in self.rows:
            for key in ["healthy_t1", "exclude_mask"]:
                if not Path(row[key]).is_file():
                    raise FileNotFoundError(row[key])

    def __len__(self):
        return len(self.rows)

    def load(self, index):
        if index in self.cache:
            self.cache.move_to_end(index)
            return self.cache[index]
        row = self.rows[index]
        im = nib.load(row["healthy_t1"])
        lm = nib.load(row["exclude_mask"])
        check_grid(im, lm)
        a, _, _ = normalize(np.asarray(im.dataobj, dtype=np.float32))
        excluded = np.asarray(lm.dataobj) > 0
        if self.margin:
            excluded = binary_dilation(excluded, iterations=self.margin)
        if (np.asarray(a.shape) < self.size).any():
            raise ValueError("MRI is smaller than the requested patch")
        fg = a > -0.95
        surface = fg & ~binary_erosion(fg, iterations=6)
        zz = np.arange(a.shape[2]) + im.affine[2, 3]
        cerebral = zz[None, None, :] > -10
        points = np.argwhere((fg & cerebral)[::4, ::4, ::4]) * 4
        cortex = np.argwhere((surface & cerebral)[::3, ::3, ::3]) * 3
        integral = np.pad(
            excluded.astype(np.int32)
            .cumsum(0, dtype=np.int32)
            .cumsum(1, dtype=np.int32)
            .cumsum(2, dtype=np.int32),
            ((1, 0),) * 3,
        )
        value = (a, im.affine.copy(), integral, points, cortex)
        self.cache[index] = value
        while len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)
        return value

    def __getitem__(self, index):
        # Any nonnegative global counter is valid, making resumes independent of worker prefetch.
        rng = np.random.default_rng(self.seed + int(index))
        scan = int(index) % len(self.rows)
        for _ in range(24):
            a, affine, s, points, cortex = self.load(scan)
            for _ in range(80):
                pool = cortex if len(cortex) and rng.random() < 0.75 else points
                if not len(pool):
                    break
                center = pool[rng.integers(len(pool))].copy()
                if rng.random() < 0.22:
                    center = center + rng.integers(-self.size // 3, self.size // 3 + 1)
                start = np.clip(
                    center - self.size // 2, 0, np.asarray(a.shape) - self.size
                )
                end = start + self.size
                x, y, z = start
                u, v, w = end
                count = (
                    s[u, v, w]
                    - s[x, v, w]
                    - s[u, y, w]
                    - s[u, v, z]
                    + s[x, y, w]
                    + s[x, v, z]
                    + s[u, y, z]
                    - s[x, y, z]
                )
                if count:
                    continue
                sl = tuple(slice(int(l), int(h)) for l, h in zip(start, end))
                patch = a[sl]
                fraction = float(np.mean(patch > -0.95))
                if fraction < 0.08:
                    continue
                origin = affine[:3, :3] @ start + affine[:3, 3]
                return {
                    "target": torch.from_numpy(patch.copy()[None]),
                    "origin": torch.tensor(origin, dtype=torch.float32),
                    "starts": torch.tensor(start),
                    "source": self.rows[scan]["healthy_t1"],
                    "site": self.rows[scan].get("site", ""),
                    "foreground_fraction": fraction,
                }
            scan = int(rng.integers(len(self.rows)))
        raise RuntimeError(
            "Could not sample a lesion-free patch; inspect the manifest and exclusion masks"
        )


class CounterSampler(Sampler):
    def __init__(self, start, count):
        self.start = start
        self.count = count

    def __len__(self):
        return self.count

    def __iter__(self):
        return iter(range(self.start, self.start + self.count))


def cortical_mask(target, origin, donors, rng, margin=8, real_only=False):
    a = target[0].numpy()
    shape = np.array(a.shape)
    fg = a > -0.95
    distance = distance_transform_edt(fg)
    variance = uniform_filter(a * a, 5) - uniform_filter(a, 5) ** 2
    band = fg & (distance >= 5) & (distance <= 28) & (variance > 0.01) & (a < 0.7)
    band &= (np.abs(np.arange(a.shape[0]) + float(origin[0])) > 20)[:, None, None]
    band &= (np.arange(a.shape[2]) + float(origin[2]))[None, None, :] > -10
    inside = np.zeros_like(fg)
    inside[tuple(slice(margin + 4, int(n) - margin - 4) for n in shape)] = True
    anchors = np.argwhere(band & inside)
    if not len(anchors):
        raise ValueError("No contextual cortical anchor")
    for _ in range(300):
        if donors and (real_only or rng.random() < 0.8):
            nii = nib.load(str(donors[int(rng.integers(len(donors)))]))
            donor = np.asarray(nii.dataobj) > 0
            loc = np.argwhere(donor)
            if not len(loc):
                continue
            donor = donor[
                tuple(slice(int(l), int(h) + 1) for l, h in zip(loc.min(0), loc.max(0)))
            ]
            donor = np.rot90(donor, int(rng.integers(4)), axes=(0, 1)).copy()
            factor = min(
                float(rng.uniform(24, 60)) / max(donor.shape),
                float((shape - 2 * margin).min()) / max(donor.shape),
            )
            donor = zoom(donor, factor, order=0) > 0
        elif real_only:
            raise ValueError("Real-mask-only sampling requires training mask donors")
        else:
            n = int(rng.integers(20, min(55, int(shape.min() - 2 * margin))))
            g = np.ogrid[tuple(slice(-1, 1, complex(n)) for _ in range(3))]
            donor = sum(v * v for v in g) < 1
        if not donor.any():
            continue
        anchor = anchors[int(rng.integers(len(anchors)))]
        start = anchor - np.array(donor.shape) // 2
        if (start < margin).any() or (start + donor.shape > shape - margin).any():
            continue
        mask = np.zeros_like(fg)
        sl = tuple(slice(int(s), int(s + n)) for s, n in zip(start, donor.shape))
        mask[sl] = donor
        if mask.sum() < 400 or np.mean(fg[mask]) < 0.9:
            continue
        cerebral = np.broadcast_to(
            (np.arange(a.shape[2]) + float(origin[2]))[None, None, :] > -10, a.shape
        )
        if np.mean(cerebral[mask]) < 0.9:
            continue
        return torch.from_numpy(mask[None].astype(np.float32))
    raise ValueError(
        "Could not place a fully contextual cortical lesion with >=90% foreground"
    )


def fixed_cases(dataset, count, donor_manifest=None, seed=1729):
    donors = []
    if donor_manifest:
        with open(donor_manifest, newline="") as f:
            donors = [r["lesion_mask"] for r in csv.DictReader(f)]
    cases = []
    seen = set()
    rng = np.random.default_rng(seed)
    for index in range(len(dataset.rows) * 30):
        case = dataset[index + 1_000_000]
        if case["source"] in seen:
            continue
        try:
            mask = cortical_mask(case["target"], case["origin"], donors, rng)
        except ValueError:
            continue
        # Derive normalization from observed full-volume values only.
        nii = nib.load(case["source"])
        raw = np.asarray(nii.dataobj, dtype=np.float32)
        missing = np.zeros(raw.shape, bool)
        starts = case["starts"].numpy()
        sl = tuple(slice(int(s), int(s + n)) for s, n in zip(starts, dataset.size))
        missing[sl] = mask[0].numpy() > 0
        normalized, _, _ = normalize(raw, missing)
        case["target"] = torch.from_numpy(normalized[sl].copy()[None])
        case["mask"] = mask
        case["image"] = case["target"] * (1 - mask)
        cases.append(case)
        seen.add(case["source"])
        if len(cases) == count:
            return cases
    raise RuntimeError(f"Could not build {count} distinct cortical validation cases")
