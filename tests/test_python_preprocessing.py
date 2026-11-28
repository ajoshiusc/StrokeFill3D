import hashlib
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import nibabel as nib
import numpy as np

from strokefill3d.normal_prior.assets import get_asset
from strokefill3d.normal_prior.medical import (
    array_from_sitk,
    n4_correct,
    register_affine,
    resample,
    to_sitk,
)
from strokefill3d.normal_prior.preprocess import prepare
from strokefill3d.normal_prior.skullstrip import conform


class AssetTests(unittest.TestCase):
    def test_download_is_verified_atomic_and_cached(self):
        payload = b"versioned preprocessing asset"
        asset = (
            "model.pt",
            "https://example.org/model.pt",
            hashlib.sha256(payload).hexdigest(),
        )
        with (
            tempfile.TemporaryDirectory() as directory,
            patch("strokefill3d.normal_prior.assets.ASSETS", {"test": asset}),
            patch(
                "strokefill3d.normal_prior.assets.urlopen",
                return_value=io.BytesIO(payload),
            ) as fetch,
        ):
            path = get_asset("test", directory)
            self.assertEqual(path.read_bytes(), payload)
            self.assertEqual(get_asset("test", directory), path)
            self.assertEqual(fetch.call_count, 1)
            path.write_bytes(b"corrupt")
            with self.assertRaisesRegex(ValueError, "checksum"):
                get_asset("test", directory)

    def test_bad_download_never_becomes_a_cached_model(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            patch(
                "strokefill3d.normal_prior.assets.ASSETS",
                {"test": ("model.pt", "https://example.org/model.pt", "bad")},
            ),
            patch(
                "strokefill3d.normal_prior.assets.urlopen",
                return_value=io.BytesIO(b"partial"),
            ),
        ):
            with self.assertRaisesRegex(ValueError, "checksum"):
                get_asset("test", directory)
            self.assertEqual(list(Path(directory).iterdir()), [])


class ConformTests(unittest.TestCase):
    def test_lia_conform_preserves_native_landmark_and_centered_padding(self):
        a = np.zeros((20, 24, 28), np.float32)
        a[3:18, 4:20, 5:25] = 1
        a[7, 11, 17] = 50
        affine = np.diag([1.0, 1.0, 1.0, 1.0])
        affine[:3, 3] = [-10, 20, -30]
        image = nib.Nifti1Image(a, affine)
        result = conform(image)
        self.assertEqual(nib.aff2axcodes(result.affine), ("L", "I", "A"))
        self.assertEqual(result.shape, (192, 192, 192))
        point = np.argwhere(np.asarray(result.dataobj) == 50)[0]
        np.testing.assert_allclose(
            nib.affines.apply_affine(result.affine, point),
            nib.affines.apply_affine(affine, [7, 11, 17]),
        )


@unittest.skipUnless(
    importlib.util.find_spec("SimpleITK"), "Install [preprocess] extra"
)
class PythonMedicalTests(unittest.TestCase):
    def setUp(self):
        import SimpleITK as sitk

        self.sitk = sitk
        self.previous_threads = sitk.ProcessObject.GetGlobalDefaultNumberOfThreads()
        sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)

    def tearDown(self):
        self.sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(self.previous_threads)

    def test_oblique_ras_lps_coordinates_and_arrays_roundtrip(self):
        theta = 0.23
        rotation = np.array(
            [
                [np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta), np.cos(theta), 0],
                [0, 0, 1],
            ]
        )
        affine = np.eye(4)
        affine[:3, :3] = rotation @ np.diag([0.8, 1.3, 2.0])
        affine[:3, 3] = [21, -43, 17]
        a = np.arange(20 * 23 * 26, dtype=np.float32).reshape(20, 23, 26)
        n = nib.Nifti1Image(a, affine)
        image = to_sitk(n)
        point = np.array([7, 11, 17])
        expected = nib.affines.apply_affine(affine, point) * [-1, -1, 1]
        np.testing.assert_allclose(
            image.TransformIndexToPhysicalPoint(tuple(map(int, point))), expected
        )
        np.testing.assert_array_equal(array_from_sitk(image), a)
        np.testing.assert_allclose(
            resample(n, n, self.sitk.Transform(3, self.sitk.sitkIdentity)), a, atol=1e-5
        )

    def test_transform_direction_labels_and_inverse(self):
        a = np.zeros((40, 44, 48), np.uint8)
        a[15:20, 20:25, 25:30] = 1
        affine = np.eye(4)
        affine[:3, 3] = [-20, -22, -24]
        n = nib.Nifti1Image(a, affine)
        transform = self.sitk.TranslationTransform(3, [3.0, -2.0, 1.0])
        result = resample(n, n, transform, True)
        # Pull in LPS: output RAS point = input RAS point + [3,-2,-1].
        expected = np.zeros_like(a)
        expected[18:23, 18:23, 24:29] = 1
        np.testing.assert_array_equal(result, expected)
        restored = resample(
            nib.Nifti1Image(result, affine), n, transform.GetInverse(), True
        )
        np.testing.assert_array_equal(restored, a)
        self.assertEqual(set(np.unique(result)), {0, 1})

    def test_n4_reduces_known_bias_and_excludes_injury(self):
        axis = np.arange(72) - 36
        x, y, z = np.meshgrid(axis, axis, axis, indexing="ij")
        brain = x * x + y * y + z * z < 30**2
        lesion = (x * x + y * y + (z - 10) ** 2) < 6**2
        observed = brain & ~lesion
        a = np.where(brain, 100 * np.exp(0.5 * x / 36), 0).astype(np.float32)
        a[lesion] = 3000
        n = nib.Nifti1Image(a, np.eye(4))
        corrected, bias, _ = n4_correct(n, observed, 1)
        before = a[observed].std() / a[observed].mean()
        after = corrected[observed].std() / corrected[observed].mean()
        self.assertLess(after, before * 0.6)
        changed = a.copy()
        changed[lesion] = 30000
        _, alternative, _ = n4_correct(nib.Nifti1Image(changed, n.affine), observed, 1)
        np.testing.assert_allclose(bias, alternative, atol=1e-5)

    def test_registration_recovers_affine_despite_excluded_injury(self):
        axis = np.arange(80) - 40
        x, y, z = np.meshgrid(axis, axis, axis, indexing="ij")
        brain = (x / 30) ** 2 + (y / 32) ** 2 + (z / 34) ** 2 < 1
        data = np.where(
            brain,
            100
            + 20 * np.sin(x / 5)
            + 15 * np.cos(y / 6)
            + 10 * np.sin(z / 4)
            + 35 * np.exp(-((x - 8) ** 2 + (y + 7) ** 2 + (z - 3) ** 2) / 100),
            0,
        ).astype(np.float32)
        affine = np.eye(4)
        affine[:3, 3] = [-40, -40, -40]
        fixed = nib.Nifti1Image(data, affine)
        angle = 0.035
        matrix = np.array(
            [
                [np.cos(angle), -np.sin(angle), 0],
                [np.sin(angle), np.cos(angle), 0],
                [0, 0, 1.0],
            ]
        ) @ np.diag([1.02, 0.99, 1.01])
        truth = self.sitk.AffineTransform(3)
        truth.SetMatrix(tuple(matrix.ravel()))
        truth.SetTranslation([2.0, -3.0, 1.5])
        moving_array = resample(fixed, fixed, truth.GetInverse())
        moving = nib.Nifti1Image(moving_array, affine)
        moving_brain = (
            resample(
                nib.Nifti1Image(brain.astype(np.uint8), affine),
                fixed,
                truth.GetInverse(),
                True,
            )
            > 0
        )
        injury = x * x + (y - 10) ** 2 + z * z < 7**2
        moving_array[injury] = 10000
        observed = moving_brain & ~injury
        estimated, _ = register_affine(fixed, moving, brain, moving_brain, observed, 1)
        landmarks = [
            (0.0, 0.0, 0.0),
            (10.0, 10.0, 10.0),
            (-15.0, 10.0, -12.0),
            (5.0, -20.0, 15.0),
        ]
        errors = [
            np.linalg.norm(
                np.array(estimated.TransformPoint(p)) - truth.TransformPoint(p)
            )
            for p in landmarks
        ]
        self.assertLess(max(errors), 1.5)
        warped = (
            resample(
                nib.Nifti1Image(moving_brain.astype(np.uint8), affine),
                fixed,
                estimated,
                True,
            )
            > 0
        )
        dice = 2 * (warped & brain).sum() / (warped.sum() + brain.sum())
        self.assertGreater(dice, 0.96)

    def test_preprocessing_uses_no_subprocess_and_records_masks_and_transforms(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shape = (64, 64, 64)
            brain = np.zeros(shape, np.uint8)
            brain[5:-5, 5:-5, 5:-5] = 1
            a = (brain * 80).astype(np.float32)
            lesion = np.zeros(shape, np.uint8)
            lesion[25:35, 25:35, 25:35] = 1
            for name, data in [("t1", a), ("brain", brain), ("lesion", lesion)]:
                nib.save(nib.Nifti1Image(data, np.eye(4)), root / f"{name}.nii.gz")
            args = SimpleNamespace(
                image=root / "t1.nii.gz",
                lesion_labels=root / "lesion.nii.gz",
                brain_mask=root / "brain.nii.gz",
                template=root / "t1.nii.gz",
                template_mask=root / "brain.nii.gz",
                output_dir=root / "prepared",
                labels=[1],
                cache_dir=None,
                threads=1,
            )

            def n4(image, observed, threads):
                self.assertFalse(observed[lesion > 0].any())
                return a, np.ones_like(a), {}

            def register(fixed, moving, fixedbrain, movingbrain, observed, threads):
                self.assertFalse(observed[lesion > 0].any())
                return self.sitk.AffineTransform(3), []

            with (
                patch(
                    "subprocess.run",
                    side_effect=AssertionError("No external executables permitted"),
                ),
                patch(
                    "strokefill3d.normal_prior.preprocess.n4_correct", side_effect=n4
                ),
                patch(
                    "strokefill3d.normal_prior.preprocess.register_affine",
                    side_effect=register,
                ),
            ):
                prepare(args)
            report = json.loads((args.output_dir / "preprocessing.json").read_text())
            self.assertEqual(report["external_executables"], [])
            self.assertEqual(report["backend"], "SimpleITK Python API")
            self.assertEqual(report["qc"]["brain_mask_dice_with_template"], 1)
            self.assertTrue((args.output_dir / "native_to_training.tfm").exists())
            np.testing.assert_array_equal(
                np.asarray(
                    nib.load(args.output_dir / "t1_brain_training.nii.gz").dataobj
                ),
                a,
            )
            with self.assertRaises(FileExistsError):
                prepare(args)

    def test_preparation_rejects_output_directory_containing_original_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "t1_native.nii.gz"
            source.write_bytes(b"original must survive")
            args = SimpleNamespace(
                output_dir=root,
                image=source,
                lesion_labels=root / "lesion.nii.gz",
                brain_mask=None,
                template=None,
                template_mask=None,
            )
            with self.assertRaisesRegex(ValueError, "preserve originals"):
                prepare(args)
            self.assertEqual(source.read_bytes(), b"original must survive")


if __name__ == "__main__":
    unittest.main()
