"""Regression checks for conditioning leakage, sampling and validation isolation."""

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from strokefill3d.anatomy import anatomy_features
from strokefill3d.model import RectifiedFlowUNet
from strokefill3d.sampling import integrate, window_plan
from strokefill3d.validation import check_split, reconstruction_metrics


class ConditioningTests(unittest.TestCase):
    def setUp(self):
        self.shape = (96, 24, 24)
        self.affine = np.eye(4)
        self.affine[:3, 3] = [-48, -12, -12]
        self.a = (
            np.random.default_rng(1).uniform(-0.8, 0.8, self.shape).astype(np.float32)
        )
        self.mask = np.zeros(self.shape, bool)
        self.mask[8:20, 8:16, 8:16] = True

    def test_hidden_target_cannot_leak_through_any_condition(self):
        before = anatomy_features(self.a, self.mask, self.affine)
        edited = self.a.copy()
        edited[self.mask] = 999
        after = anatomy_features(edited, self.mask, self.affine)
        torch.testing.assert_close(before, after, rtol=0, atol=0)

    def test_real_excluded_lesion_cannot_leak(self):
        excluded = np.zeros(self.shape, bool)
        excluded[70:80, 6:18, 6:18] = True
        before = anatomy_features(self.a, self.mask, self.affine, excluded=excluded)
        edited = self.a.copy()
        edited[excluded] = -100
        after = anatomy_features(edited, self.mask, self.affine, excluded=excluded)
        torch.testing.assert_close(before, after, rtol=0, atol=0)

    def test_mirror_uses_world_midline_not_array_midpoint(self):
        a = np.zeros_like(self.a)
        a[18, 12, 12] = 0.75
        features = anatomy_features(a, np.zeros_like(self.mask), self.affine)
        # World x=-30 reflects to +30, voxel 78 (not index 77).
        self.assertAlmostEqual(float(features[0, 78, 12, 12]), 0.75)
        self.assertAlmostEqual(float(features[2, 78, 12, 12]), 0.30, places=6)

    def test_low_intensity_csf_is_available_to_mirror(self):
        a = np.full(self.shape, -1.0, dtype=np.float32)
        a[18, 12, 12] = -0.98
        features = anatomy_features(a, np.zeros_like(self.mask), self.affine)
        self.assertEqual(float(features[1, 78, 12, 12]), 1.0)
        self.assertAlmostEqual(float(features[0, 78, 12, 12]), -0.98, places=6)

    def test_crop_matches_full_features(self):
        crop = (slice(3, 19), slice(4, 20), slice(2, 18))
        full = anatomy_features(self.a, self.mask, self.affine)
        part = anatomy_features(self.a, self.mask, self.affine, crop)
        torch.testing.assert_close(part, full[(slice(None), *crop)])


class PipelineTests(unittest.TestCase):
    def test_window_plan_rejects_unsafe_centered_roi(self):
        m = np.zeros((40, 40, 40), bool)
        m[2:38, 15:25, 15:25] = True
        with self.assertRaises(ValueError):
            window_plan(m, (32,) * 3, "centered")
        _, method = window_plan(m, (32,) * 3, "auto")
        self.assertEqual(method, "gaussian")

    def test_sparse_gaussian_preserves_observed_voxels(self):
        class Constant(torch.nn.Module):
            def predict_input(self, x, t):
                return torch.ones_like(x[:, :1])

        context = torch.randn(1, 1, 24, 24, 24)
        mask = torch.zeros_like(context)
        mask[:, :, 5:19, 5:19, 5:19] = 1
        noise = torch.zeros_like(context)
        output, _ = integrate(
            Constant(),
            context,
            mask,
            None,
            noise,
            (16,) * 3,
            steps=3,
            strategy="gaussian",
        )
        torch.testing.assert_close(
            output[mask == 0], context[mask == 0], rtol=0, atol=0
        )
        torch.testing.assert_close(output[mask > 0], torch.ones_like(output[mask > 0]))

    def test_anatomical_model_checkpoint_backward(self):
        model = RectifiedFlowUNet(
            channels=(8, 16),
            attention_levels=(False, True),
            num_head_channels=(0, 8),
            anatomy_channels=7,
            use_flash_attention=True,
            activation_checkpointing=True,
        )
        x = torch.randn(1, 1, 16, 16, 16)
        condition = torch.randn(1, 7, 16, 16, 16)
        y = model(x, x, torch.ones_like(x), torch.tensor([500.0]), condition)
        y.square().mean().backward()
        self.assertTrue(
            all(
                torch.isfinite(p.grad).all()
                for p in model.parameters()
                if p.grad is not None
            )
        )
        with self.assertRaises(ValueError):
            model(x, x, torch.ones_like(x), torch.tensor([0.0]))

    def test_validation_metrics_reward_exact_target(self):
        torch.manual_seed(2)
        target = torch.rand(1, 1, 20, 20, 20) * 2 - 1
        mask = torch.zeros_like(target)
        mask[:, :, 5:15, 5:15, 5:15] = 1
        exact = reconstruction_metrics(target, target, mask)
        noisy = reconstruction_metrics(
            target + 0.2 * torch.randn_like(target) * mask, target, mask
        )
        self.assertEqual(exact["masked_mae"], 0)
        self.assertAlmostEqual(exact["bbox_ssim"], 1.0, places=5)
        self.assertGreater(noisy["masked_mae"], 0)
        self.assertLess(noisy["bbox_ssim"], exact["bbox_ssim"])

    def test_masked_ssim_scores_missing_region_and_handles_face_masks(self):
        from monai.metrics.regression import compute_ssim_and_cs

        for dims in (2, 3):
            torch.manual_seed(25)
            target = torch.rand(1, 1, *([24] * dims)) * 2 - 1
            mask = torch.zeros_like(target)
            mask[(slice(None), slice(None), *[slice(8, 10)] * dims)] = 1
            mask[(slice(None), slice(None), *[slice(14, 16)] * dims)] = 1
            pred = target * (1 - mask)
            result = reconstruction_metrics(pred, target, mask)
            padding = [3, 3] * dims
            similarity, _ = compute_ssim_and_cs(
                torch.nn.functional.pad(pred, padding, mode="reflect"),
                torch.nn.functional.pad(target, padding, mode="reflect"),
                dims,
                (7,) * dims,
                (1.5,) * dims,
                data_range=2.0,
            )
            expected = float((similarity * mask).sum() / mask.sum())
            self.assertAlmostEqual(result["masked_ssim"], expected, places=6)
            self.assertLess(result["masked_ssim"], result["bbox_ssim"])
            face_mask = torch.zeros_like(target)
            face_mask[(slice(None), slice(None), *[slice(0, 1)] * dims)] = 1
            face_result = reconstruction_metrics(target, target, face_mask)
            self.assertAlmostEqual(face_result["masked_ssim"], 1.0, places=5)

    def test_resume_does_not_compare_incompatible_validation_scores(self):
        from strokefill3d.train import compatible_best_score

        self.assertEqual(
            compatible_best_score({"best_score": 0.1, "metrics": {"bbox_ssim": 0.9}}),
            float("inf"),
        )
        self.assertEqual(
            compatible_best_score({"best_score": 0.4, "metrics": {"masked_ssim": 0.5}}),
            0.4,
        )
        self.assertEqual(
            compatible_best_score({"best_score": 0.4, "validation_metric_version": 2}),
            0.4,
        )

    def test_site_and_donor_leakage_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            def write(name, rows):
                p = root / name
                with p.open("w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=list(rows[0]))
                    w.writeheader()
                    w.writerows(rows)
                return p

            a = write(
                "train.csv",
                [
                    {
                        "healthy_t1": "/a/sub-1/ses-1/anat/t1.nii.gz",
                        "exclude_mask": "/a/mask.nii.gz",
                        "site": "A",
                    }
                ],
            )
            b = write(
                "val.csv",
                [
                    {
                        "healthy_t1": "/b/sub-2/ses-1/anat/t1.nii.gz",
                        "exclude_mask": "/b/mask.nii.gz",
                        "site": "A",
                    }
                ],
            )
            with self.assertRaisesRegex(ValueError, "Site"):
                check_split(a, b)
            b = write(
                "val.csv",
                [
                    {
                        "healthy_t1": "/b/sub-2/ses-1/anat/t1.nii.gz",
                        "exclude_mask": "/b/mask.nii.gz",
                        "site": "B",
                    }
                ],
            )
            donors = write("donors.csv", [{"lesion_mask": "/b/mask.nii.gz"}])
            with self.assertRaisesRegex(ValueError, "Donor"):
                check_split(a, b, donors)


if __name__ == "__main__":
    unittest.main()
