import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from strokefill3d.normal_prior.core import file_hash, load_checkpoint_snapshot


class CheckpointSnapshotTests(unittest.TestCase):
    def test_wrong_fingerprint_is_rejected_before_deserializing(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "codec.pt"
            path.write_bytes(b"wrong codec")
            with patch("strokefill3d.normal_prior.core.torch.load") as load:
                with self.assertRaisesRegex(ValueError, "fingerprints"):
                    load_checkpoint_snapshot(path, expected_sha256="wrong")
                load.assert_not_called()

    def test_atomic_replacement_cannot_mismatch_loaded_weights_and_fingerprint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            current, replacement = root / "best.pt", root / "next.pt"
            torch.save({"step": 10, "model": torch.tensor([10.0])}, current)
            torch.save({"step": 20, "model": torch.tensor([20.0])}, replacement)
            old_hash = file_hash(current)
            torch_load = torch.load

            def publish_next_then_load(stream, **kwargs):
                # Training publishes its next checkpoint after the hash is read,
                # while inference still owns the previous checkpoint's descriptor.
                replacement.replace(current)
                return torch_load(stream, **kwargs)

            with patch(
                "strokefill3d.normal_prior.core.torch.load",
                side_effect=publish_next_then_load,
            ):
                saved, digest = load_checkpoint_snapshot(current)
            self.assertEqual(saved["step"], 10)
            self.assertEqual(saved["model"].item(), 10)
            self.assertEqual(digest, old_hash)
            self.assertNotEqual(digest, file_hash(current))
            self.assertEqual(torch_load(current, weights_only=True)["step"], 20)


if __name__ == "__main__":
    unittest.main()
