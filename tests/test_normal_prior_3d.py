"""Normal-prior invariants: hidden-data isolation, differentiable RF and run safety."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
import torch.nn.functional as F

from strokefill3d.normal_prior.core import (
    VERSION,
    coordinate_maps,
    fit_source,
    load_bundle,
    solve,
)
from strokefill3d.normal_prior.data import normalize
from strokefill3d.normal_prior.infer import centered_roi
from strokefill3d.normal_prior.io import (
    RunLock,
    atomic_json,
    atomic_save,
    initialize_run,
)
from strokefill3d.normal_prior.train import CodeCache, ShardSampler


class BlockCodec:
    patch_size = (8, 8, 8)
    channels = 1
    factor = 2

    def decode(self, code):
        return F.interpolate(code, scale_factor=2, mode="nearest")


class ZeroVelocity(torch.nn.Module):
    def forward(self, x, time):
        return x[:, :1] * 0


class NormalPriorTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_world_coordinates_are_block_centers(self):
        origins = torch.tensor([[-98.0, -134.0, -72.0], [-60.0, -50.0, 0.0]])
        full = coordinate_maps(origins, [8] * 3, 1)
        pooled = F.avg_pool3d(full, 2)
        torch.testing.assert_close(coordinate_maps(origins, [4] * 3, 2), pooled)

    def test_flow_direction_and_checkpointed_source_gradient(self):
        class Constant(torch.nn.Module):
            def forward(self, x, time):
                return torch.ones_like(x[:, :1]) * 2

        source = torch.randn(1, 1, 4, 4, 4, requires_grad=True)
        out = solve(Constant(), source, torch.zeros(1, 3, 4, 4, 4), 4)
        torch.testing.assert_close(out, source + 2)
        out.sum().backward()
        torch.testing.assert_close(source.grad, torch.ones_like(source))

    def test_source_fit_reduces_visible_error_and_preserves_observations(self):
        image = torch.full((1, 1, 8, 8, 8), 0.3)
        mask = torch.zeros_like(image)
        mask[:, :, 2:6, 2:6, 2:6] = 1
        image[mask > 0] = 0
        coords = torch.zeros(1, 3, 4, 4, 4)
        out, code, source, history = fit_source(
            ZeroVelocity(), BlockCodec(), image, mask, coords, iterations=40, steps=2
        )
        self.assertLess(history[-1]["observed_mse"], history[0]["observed_mse"] * 0.1)
        torch.testing.assert_close(out[mask == 0], image[mask == 0], rtol=0, atol=0)
        self.assertTrue(torch.isfinite(source).all())
        # Error above is scored BEFORE observed pixels are copied into the output.
        self.assertEqual(code.shape, (1, 1, 4, 4, 4))

    def test_hidden_inputs_rejected_and_normalization_cannot_leak(self):
        a = np.random.default_rng(8).uniform(10, 90, (16,) * 3).astype(np.float32)
        missing = np.zeros_like(a, bool)
        missing[4:9, 4:9, 4:9] = True
        before, lo, hi = normalize(a, missing)
        a[missing] = np.nan
        after, lo2, hi2 = normalize(a, missing)
        np.testing.assert_array_equal(before[~missing], after[~missing])
        self.assertEqual((lo, hi), (lo2, hi2))
        image = torch.ones(1, 1, 8, 8, 8)
        mask = torch.ones_like(image)
        with self.assertRaisesRegex(ValueError, "Hidden MRI"):
            fit_source(
                ZeroVelocity(), BlockCodec(), image, mask, torch.zeros(1, 3, 4, 4, 4)
            )

    def test_no_silent_tiling_for_oversized_components(self):
        mask = np.zeros((120,) * 3, bool)
        mask[5:105, 40:60, 40:60] = True
        with self.assertRaisesRegex(ValueError, "does not fit"):
            centered_roi(mask, mask.shape, (96,) * 3)

    def test_legacy_and_mismatched_checkpoints_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder)
            (p / "prior").mkdir()
            atomic_save({"kind": "old_pixel_rf"}, p / "prior/best.pt")
            with self.assertRaisesRegex(ValueError, "Not a normal-prior"):
                load_bundle(p, device="cpu")
            (p / "codec.pt").write_bytes(b"wrong codec")
            atomic_save(
                {"kind": VERSION + "_prior", "codec_sha256": "wrong"},
                p / "prior/best.pt",
            )
            with self.assertRaisesRegex(ValueError, "fingerprints"):
                load_bundle(p, device="cpu")

    def test_cache_fingerprints_and_corruption_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder)
            atomic_json(
                {
                    "codec_sha256": "A",
                    "train_sha256": "B",
                    "shards": [{"name": "000.pt", "count": 2, "sha256": "bad"}],
                },
                p / "complete.json",
            )
            with self.assertRaisesRegex(ValueError, "fingerprints"):
                CodeCache(p, "C", "B")
            (p / "000.pt").write_bytes(b"corrupted")
            with self.assertRaisesRegex(ValueError, "Corrupt"):
                CodeCache(p, "A", "B")[0]

    def test_shard_sampler_resume_does_not_repeat_or_skip(self):
        class Data:
            def __init__(self):
                self.meta = {"shards": [{}, {}, {}]}
                self.offsets = np.array([0, 4, 8, 12])

            def __len__(self):
                return 12

        d = Data()
        whole = list(ShardSampler(d, 0, 29, 91))
        self.assertEqual(sorted(whole[:12]), list(range(12)))
        self.assertEqual(whole[7:], list(ShardSampler(d, 7, 22, 91)))
        self.assertEqual(len({x // 4 for x in whole[:4]}), 1)

    def test_atomic_failure_retains_previous_checkpoint(self):
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder) / "ck.pt"
            atomic_save({"step": 4}, p)
            with (
                patch("torch.save", side_effect=OSError("disk full")),
                self.assertRaises(OSError),
            ):
                atomic_save({"step": 5}, p)
            self.assertEqual(torch.load(p, weights_only=True)["step"], 4)

    def test_run_lock_and_manifest_changes_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder)
            manifest = p / "train.csv"
            manifest.write_text("v1")
            run = p / "run"
            with RunLock(run):
                with (
                    self.assertRaisesRegex(RuntimeError, "Another process"),
                    RunLock(run),
                ):
                    pass
                identity = initialize_run(
                    run, {"patch": 96}, {"train": manifest}, False
                )
                self.assertEqual(identity["version"], VERSION)
                with self.assertRaisesRegex(ValueError, "already exists"):
                    initialize_run(run, {"patch": 96}, {"train": manifest}, False)
                manifest.write_text("v2")
                with self.assertRaisesRegex(ValueError, "differ"):
                    initialize_run(run, {"patch": 96}, {"train": manifest}, True)


if __name__ == "__main__":
    unittest.main()
