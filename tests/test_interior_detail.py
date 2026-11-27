import unittest

import torch
import torch.nn.functional as F

from strokefill3d.validation import interior_detail_metrics


class InteriorDetailTests(unittest.TestCase):
    def test_interior_does_not_reward_copied_surroundings(self):
        torch.manual_seed(4)
        target = torch.rand(1, 1, 32, 32) * 2 - 1
        mask = torch.zeros_like(target)
        mask[:, :, 8:24, 8:24] = 1
        pred = torch.where(mask > 0, torch.zeros_like(target), target)
        first = interior_detail_metrics(pred, target, mask)
        altered = torch.where(mask > 0, pred, torch.ones_like(pred) * -1)
        second = interior_detail_metrics(altered, target, mask)
        self.assertEqual(first, second)
        self.assertEqual(first["interior_voxels"], 100)
        self.assertLess(first["interior_gradient_ratio"], 0.01)
        self.assertLess(first["interior_ssim"], 0.1)

    def test_aligned_details_differ_from_noise_and_blur_in_2d_and_3d(self):
        torch.manual_seed(8)
        for dims in (2, 3):
            shape = (1, 1) + (20,) * dims
            target = torch.rand(shape) * 2 - 1
            mask = torch.ones_like(target)
            exact = interior_detail_metrics(target, target, mask)
            pool = F.avg_pool2d if dims == 2 else F.avg_pool3d
            smooth = interior_detail_metrics(pool(target, 3, 1, 1), target, mask)
            noisy = interior_detail_metrics(
                torch.rand_like(target) * 2 - 1, target, mask
            )
            self.assertAlmostEqual(exact["interior_ssim"], 1, places=5)
            self.assertAlmostEqual(exact["interior_gradient_cosine"], 1, places=5)
            self.assertAlmostEqual(exact["interior_gradient_ratio"], 1, places=5)
            self.assertLess(smooth["interior_gradient_ratio"], 0.3)
            self.assertGreater(noisy["interior_gradient_ratio"], 0.7)
            self.assertLess(abs(noisy["interior_gradient_cosine"]), 0.15)

    def test_thin_or_constant_regions_have_explicit_undefined_scores(self):
        image = torch.zeros(1, 1, 20, 20, 20)
        mask = torch.zeros_like(image)
        mask[:, :, 4:6, 4:16, 4:16] = 1
        thin = interior_detail_metrics(image, image, mask)
        self.assertEqual(thin["interior_voxels"], 0)
        self.assertIsNone(thin["interior_ssim"])
        flat = interior_detail_metrics(image, image, torch.ones_like(mask))
        self.assertIsNone(flat["interior_gradient_ratio"])
        self.assertIsNone(flat["interior_gradient_cosine"])
        self.assertEqual(flat["interior_gradient_mae"], 0)


if __name__ == "__main__":
    unittest.main()
