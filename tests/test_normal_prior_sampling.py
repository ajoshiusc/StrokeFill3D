"""A complete training target must exclude every voxel of the real lesion margin."""

import csv
import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from scipy.ndimage import binary_dilation

from strokefill3d.normal_prior.data import NormalPatches


class SamplingTests(unittest.TestCase):
    def test_stateless_clean_targets_with_full_3d_exclusion(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            grid = np.stack(np.meshgrid(*[np.arange(80)] * 3, indexing="ij"))
            brain = ((grid - 40) ** 2).sum(0) < 34**2
            image = np.where(
                brain, 70 + 20 * np.cos(grid[0] / 3) * np.cos(grid[2] / 3), 0
            ).astype(np.float32)
            lesion = np.zeros((80,) * 3, np.uint8)
            lesion[34:46, 34:46, 34:46] = 1
            excluded = binary_dilation(lesion, iterations=3)
            affine = np.eye(4)
            affine[:3, 3] = [-40, -40, -30]
            nib.save(nib.Nifti1Image(image, affine), root / "t1.nii.gz")
            nib.save(nib.Nifti1Image(lesion, affine), root / "lesion.nii.gz")
            manifest = root / "train.csv"
            with manifest.open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["healthy_t1", "exclude_mask", "site"])
                w.writeheader()
                w.writerow(
                    {
                        "healthy_t1": str(root / "t1.nii.gz"),
                        "exclude_mask": str(root / "lesion.nii.gz"),
                        "site": "synthetic",
                    }
                )
            dataset = NormalPatches(
                manifest, (32,) * 3, seed=42, exclusion_margin=3, cache_size=1
            )
            starts = []
            for index in range(8):
                a = dataset[index]
                b = dataset[index]
                torch.testing.assert_close(a["target"], b["target"], rtol=0, atol=0)
                torch.testing.assert_close(a["starts"], b["starts"], rtol=0, atol=0)
                sl = tuple(slice(int(s), int(s + 32)) for s in a["starts"])
                self.assertFalse(excluded[sl].any())
                self.assertEqual(a["target"].shape, (1, 32, 32, 32))
                self.assertGreaterEqual(a["foreground_fraction"], 0.08)
                starts.append(tuple(a["starts"].tolist()))
            self.assertGreater(len(set(starts)), 1)


if __name__ == "__main__":
    unittest.main()
