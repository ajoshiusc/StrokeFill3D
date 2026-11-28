"""Brain-normalized targets, whole-patch lesion exclusion, balanced domains."""

import csv
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from scipy.ndimage import binary_erosion, distance_transform_edt
from torch.utils.data import Dataset

from strokefill3d.normal_prior.data import NormalPatches, check_grid, cortical_mask


def read_rows(path):
    with open(path, newline="") as stream:
        return list(csv.DictReader(stream))


def observed_validation_target(case, rows):
    """Exclude synthetic hidden voxels when deriving validation intensity scale."""
    row = next(r for r in rows if r["healthy_t1"] == case["source"])
    raw = np.asarray(nib.load(row["healthy_t1"]).dataobj, dtype=np.float32)
    brain = (
        np.asarray(nib.load(row["brain_mask"]).dataobj) > 0
        if row.get("brain_mask")
        else raw > 0
    )
    lesion = np.asarray(nib.load(row["exclude_mask"]).dataobj) > 0
    valid = brain & (distance_transform_edt(~lesion) > 5)
    shape = case["target"].shape[1:]
    sl = tuple(slice(int(s), int(s + n)) for s, n in zip(case["starts"], shape))
    local = valid[sl]
    local[case["mask"][0].numpy() > 0] = False
    lo, hi = np.percentile(raw[valid], [1, 99])
    patch = np.clip(2 * (raw[sl] - lo) / (hi - lo) - 1, -1, 1).astype(np.float32)
    patch[~brain[sl]] = -1
    return torch.from_numpy(patch[None])


class CleanDomain(NormalPatches):
    def load(self, index):
        if index in self.cache:
            self.cache.move_to_end(index)
            return self.cache[index]
        row = self.rows[index]
        im, lm = nib.load(row["healthy_t1"]), nib.load(row["exclude_mask"])
        check_grid(im, lm)
        expected = np.eye(4)
        expected[:3, 3] = [-98, -134, -72]
        if im.shape != (197, 233, 189) or not np.allclose(
            im.affine, expected, atol=1e-3
        ):
            raise ValueError("Mixed training requires the canonical 1-mm MNI grid")
        raw = np.asarray(im.dataobj, dtype=np.float32)
        lesion = np.asarray(lm.dataobj) > 0
        if row.get("brain_mask"):
            bm = nib.load(row["brain_mask"])
            check_grid(im, bm)
            brain = np.asarray(bm.dataobj) > 0
        else:
            brain = raw > 0
        if not np.isfinite(raw).all():
            raise ValueError("Nonfinite T1")
        excluded = distance_transform_edt(~lesion) <= self.margin
        observed = brain & ~excluded
        if observed.sum() < 10000:
            raise ValueError("Too little observed brain")
        lo, hi = np.percentile(raw[observed], [1, 99])
        if hi <= lo:
            raise ValueError("Degenerate intensity range")
        a = np.clip(2 * (raw - lo) / (hi - lo) - 1, -1, 1).astype(np.float32)
        a[~brain] = -1
        surface = brain & ~binary_erosion(brain, iterations=6)
        cerebral = (np.arange(a.shape[2]) + im.affine[2, 3])[None, None, :] > -10
        points = np.argwhere((brain & cerebral)[::4, ::4, ::4]) * 4
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


class MixedPatches(Dataset):
    def __init__(self, manifest, patch_size, seed=2026, cache_size=2):
        rows = read_rows(manifest)
        self.seed = seed
        self.domains = sorted({r["domain"] for r in rows})
        self.datasets = {}
        for domain in self.domains:
            ds = CleanDomain(manifest, patch_size, seed, 5, cache_size)
            ds.rows = [r for r in ds.rows if r["domain"] == domain]
            self.datasets[domain] = ds
        # Half ATLAS; other half shared equally among external domains.
        self.weights = np.array(
            [len(self.domains) - 1 if d == "atlas" else 1 for d in self.domains], float
        )
        self.weights /= self.weights.sum()
        self.donors = [r["exclude_mask"] for r in rows]
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        rng = np.random.default_rng(self.seed + int(index))
        domain = self.domains[int(rng.choice(len(self.domains), p=self.weights))]
        ds = self.datasets[domain]
        for attempt in range(32):
            case = ds[int(index) + attempt * 7919]
            try:
                mask = cortical_mask(
                    case["target"], case["origin"], self.donors, rng, real_only=True
                )
            except ValueError:
                continue
            case.update(mask=mask, domain=domain)
            return case
        raise RuntimeError(
            f"Cannot place real cortical masks in {domain} clean patches"
        )


def validate_splits(train, val, test):
    groups = [read_rows(p) for p in (train, val, test)]
    required = {"healthy_t1", "exclude_mask", "subject", "domain"}
    for rows in groups:
        if not rows or not required.issubset(rows[0]):
            raise ValueError("Empty or invalid manifest")
        for r in rows:
            for k in ["healthy_t1", "exclude_mask"] + (
                ["brain_mask"] if r.get("brain_mask") else []
            ):
                if not Path(r[k]).is_file():
                    raise FileNotFoundError(r[k])
            if "inpaint" in Path(r["healthy_t1"]).name.lower():
                raise ValueError("Generated MRI cannot be a target")
    for i, a in enumerate(groups):
        for b in groups[i + 1 :]:
            for key in ["subject", "healthy_t1", "exclude_mask"]:
                if {r[key] for r in a} & {r[key] for r in b}:
                    raise ValueError(f"{key} leakage across splits")
    return groups
