"""Regression checks for clean-target RF domain splits and observation locking."""

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from strokefill3d.mixed_rf.core import sample, velocity
from strokefill3d.mixed_rf.data import validate_splits
from strokefill3d.normal_prior.data import cortical_mask


class Endpoint(torch.nn.Module):
    def forward(self, x, t):
        return torch.zeros_like(x[:, :1]) + 0.25


class MixedRFTests(unittest.TestCase):
    def test_clean_head_velocity_and_observation_lock(self):
        image = torch.full((1, 1, 16, 16, 16), -0.2)
        mask = torch.zeros_like(image)
        mask[:, :, 4:12, 4:12, 4:12] = 1
        image = torch.where(mask > 0, 0, image)
        out = sample(Endpoint(), image, mask, torch.zeros(1, 3), seed=9)
        self.assertTrue(torch.equal(out[mask == 0], image[mask == 0]))
        self.assertTrue(
            torch.allclose(
                out[mask > 0], torch.full_like(out[mask > 0], 0.25), atol=1e-5
            )
        )
        v = velocity(
            Endpoint(),
            torch.zeros_like(image),
            image,
            mask,
            torch.zeros(1, 3, 16, 16, 16),
            0.5,
        )
        self.assertTrue(torch.allclose(v, torch.full_like(v, 0.5)))

    def test_real_only_never_falls_back_to_sphere(self):
        with self.assertRaises(ValueError):
            cortical_mask(
                torch.rand(1, 64, 64, 64),
                torch.tensor([20, 0, 0]),
                [],
                np.random.default_rng(1),
                real_only=True,
            )

    def test_subject_leakage_rejected_even_when_paths_differ(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            files = []
            for i in range(3):
                source = root / f"t1_{i}.nii"
                source.touch()
                mask = root / f"mask_{i}.nii"
                mask.touch()
                manifest = root / f"{i}.csv"
                with manifest.open("w", newline="") as f:
                    w = csv.DictWriter(
                        f,
                        fieldnames=["healthy_t1", "exclude_mask", "subject", "domain"],
                    )
                    w.writeheader()
                    w.writerow(
                        {
                            "healthy_t1": source,
                            "exclude_mask": mask,
                            "subject": "same" if i < 2 else "third",
                            "domain": "arc",
                        }
                    )
                files.append(manifest)
            with self.assertRaisesRegex(ValueError, "subject leakage"):
                validate_splits(*files)


if __name__ == "__main__":
    unittest.main()
