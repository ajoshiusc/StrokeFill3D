"""Prepare an external T1 and aligned injury mask entirely inside Python.

Install pip extra [preprocess]. Uses SimpleITK for N4/global/affine registration
and PyTorch SynthStrip for brain extraction. Default assets download on first
use with checksum validation; explicit paths support offline operation.
"""

import argparse
import json
import time
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import binary_dilation

from .assets import get_asset, sha256
from .io import RunLock, atomic_json
from .medical import n4_correct, register_affine, resample, sitk_module

# Retain these public helpers for existing callers.
fingerprint = sha256


def save(data, reference, path, dtype=np.float32):
    header = reference.header.copy()
    header.set_data_dtype(dtype)
    nib.save(
        nib.Nifti1Image(np.asarray(data, dtype=dtype), reference.affine, header), path
    )


def canonical(path):
    image = nib.load(path)
    data = np.asarray(image.dataobj)
    if data.ndim > 3 and all(v == 1 for v in data.shape[3:]):
        image = nib.Nifti1Image(
            data.reshape(data.shape[:3]), image.affine, image.header
        )
    if len(image.shape) != 3 or not np.isfinite(np.asarray(image.dataobj)).all():
        raise ValueError(f"Require a finite 3D NIfTI: {path}")
    return nib.as_closest_canonical(image)


def aligned(a, b):
    if a.shape != b.shape or not np.allclose(a.affine, b.affine, atol=1e-4):
        raise ValueError(
            "Input T1 and labels/mask do not share a grid. Supply verified, aligned masks."
        )


def prepare(args):
    sitk = sitk_module()
    out = Path(args.output_dir).resolve()
    inputs = [
        args.image,
        args.lesion_labels,
        args.brain_mask,
        args.template,
        args.template_mask,
    ]
    if any(out in Path(path).resolve().parents for path in inputs if path):
        raise ValueError("Keep input files outside --output-dir to preserve originals")
    out.mkdir(parents=True, exist_ok=True)
    with RunLock(out):
        if (out / "preprocessing.json").exists():
            raise FileExistsError(
                "Use a new output directory; completed preprocessing will not be overwritten"
            )
        started = time.monotonic()
        native = canonical(args.image)
        labels = canonical(args.lesion_labels)
        aligned(native, labels)
        raw = np.asarray(native.dataobj, dtype=np.float32)
        label_data = np.asarray(labels.dataobj)
        if not np.allclose(label_data, np.rint(label_data)):
            raise ValueError("Lesion labels must be discrete integers")
        lesion = np.isin(label_data, args.labels)
        if not lesion.any():
            raise ValueError("Requested lesion labels are empty")
        template = args.template or get_asset("template", args.cache_dir)
        template_mask = args.template_mask or get_asset("template_mask", args.cache_dir)
        fixed = canonical(template)
        fixedmask = canonical(template_mask)
        aligned(fixed, fixedmask)
        if not np.allclose(fixed.affine[:3, :3], np.eye(3), atol=1e-3):
            raise ValueError("Template must have an RAS 1-mm grid")
        fixedbrain = np.asarray(fixedmask.dataobj) > 0
        if fixedbrain.sum() < 10000:
            raise ValueError("Template brain mask is empty or too small")
        save(raw, native, out / "t1_native.nii.gz")
        save(lesion, native, out / "lesion_native.nii.gz", np.uint8)
        if args.brain_mask:
            masknii = canonical(args.brain_mask)
            aligned(native, masknii)
            brain = np.asarray(masknii.dataobj) > 0
            extraction = {
                "method": "provided_mask",
                "mask_sha256": sha256(args.brain_mask),
            }
        else:
            from .skullstrip import extract_brain

            print("Brain extraction: SynthStrip in PyTorch", flush=True)
            brain, extraction = extract_brain(
                native,
                args.synthstrip_weights,
                args.cache_dir,
                args.brain_device,
                args.threads,
            )
        save(brain, native, out / "brain_mask_native.nii.gz", np.uint8)
        observed = brain & ~binary_dilation(lesion, iterations=3) & (raw > 0)
        if observed.sum() < 10000:
            raise ValueError(
                "Insufficient observed brain for bias estimation and registration"
            )
        save(observed, native, out / "observed_brain_native.nii.gz", np.uint8)
        print("N4 bias correction: SimpleITK", flush=True)
        corrected, bias, n4_report = n4_correct(native, observed, args.threads)
        save(corrected, native, out / "t1_n4_native.nii.gz")
        save(bias, native, out / "bias_native.nii.gz")
        save(np.where(brain, corrected, 0), native, out / "brain_n4_native.nii.gz")
        save(
            np.where(fixedbrain, np.asarray(fixed.dataobj), 0),
            fixed,
            out / "template_brain.nii.gz",
        )
        save(fixedbrain, fixed, out / "template_mask.nii.gz", np.uint8)
        moving = nib.Nifti1Image(corrected, native.affine)
        transform, stages = register_affine(
            fixed, moving, fixedbrain, brain, observed, args.threads
        )
        matrix = np.asarray(transform.GetMatrix()).reshape(3, 3)
        determinant = float(np.linalg.det(matrix))
        scales = np.linalg.svd(matrix, compute_uv=False)
        if (
            not np.isfinite(matrix).all()
            or determinant <= 0
            or scales.min() < 0.5
            or scales.max() > 2
        ):
            raise RuntimeError(
                "Registration affine has reflection or implausible scales"
            )
        # SimpleITK resampling maps output/template LPS points into native LPS.
        sitk.WriteTransform(transform, str(out / "training_to_native.tfm"))
        sitk.WriteTransform(transform.GetInverse(), str(out / "native_to_training.tfm"))
        a = resample(moving, fixed, transform)
        bm = (
            resample(
                nib.Nifti1Image(brain.astype(np.uint8), native.affine),
                fixed,
                transform,
                True,
            )
            > 0
        )
        lm = (
            resample(
                nib.Nifti1Image(lesion.astype(np.uint8), native.affine),
                fixed,
                transform,
                True,
            )
            > 0
        )
        save(a, fixed, out / "t1_training_fullhead.nii.gz")
        save(bm, fixed, out / "brain_mask_training.nii.gz", np.uint8)
        save(lm, fixed, out / "lesion_training.nii.gz", np.uint8)
        save(np.where(bm, a, 0), fixed, out / "t1_brain_training.nii.gz")
        dice = (
            2 * int((bm & fixedbrain).sum()) / max(1, int(bm.sum() + fixedbrain.sum()))
        )
        valid = bm & ~lm
        if not valid.any():
            raise RuntimeError("No observed brain remains after registration")
        metrics = {
            "brain_mask_dice_with_template": dice,
            "brain_volume_ml": float(bm.sum() / 1000),
            "lesion_voxels": int(lm.sum()),
            "lesion_voxels_outside_brain": int((lm & ~bm).sum()),
            "nonpositive_jacobian_voxels_in_brain": 0,
            "affine_matrix_determinant": determinant,
            "affine_matrix_singular_values": scales.tolist(),
            "observed_brain_percentiles": np.percentile(
                a[valid], [1, 10, 50, 90, 99]
            ).tolist(),
            "visual_review_required": True,
        }
        manifest = {
            "preprocessing_version": 2,
            "backend": "SimpleITK Python API",
            "backend_version": sitk.Version_VersionString(),
            "implementation_sha256": {
                name: sha256(Path(__file__).with_name(name))
                for name in [
                    "preprocess.py",
                    "medical.py",
                    "skullstrip.py",
                    "assets.py",
                ]
            },
            "template_space": "MNI152NLin2009aSym (supplied or cached template)",
            "registration": "affine",
            "input_files": {
                str(Path(x).resolve()): sha256(x)
                for x in [args.image, args.lesion_labels, template, template_mask]
                + ([args.brain_mask] if args.brain_mask else [])
            },
            "brain_extraction": extraction,
            "bias_correction": {
                "method": "N4",
                "injury_exclusion_margin_voxels": 3,
                **n4_report,
            },
            "registration_stages": stages,
            "transforms": {
                "training_to_native.tfm": "template LPS mm -> native LPS mm (pull for MNI resampling)",
                "native_to_training.tfm": "native LPS mm -> template LPS mm (pull for native resampling)",
            },
            "intensity_mapping": "none; inference applies training p1/p99 scaling to skull-stripped observed voxels",
            "external_executables": [],
            "threads": args.threads,
            "elapsed_seconds": time.monotonic() - started,
            "qc": metrics,
        }
        atomic_json(manifest, out / "preprocessing.json")
        print(json.dumps(metrics, indent=2), flush=True)
        if dice < 0.8:
            raise RuntimeError(
                "Registration QC failed; inspect outputs before inference"
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["image", "lesion-labels", "output-dir"]:
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--labels", nargs="+", type=int, default=[1, 2])
    parser.add_argument(
        "--template",
        help="Exact MNI152NLin2009aSym RAS 1-mm T1; cached default if omitted",
    )
    parser.add_argument("--template-mask", help="Corresponding template brain mask")
    parser.add_argument(
        "--cache-dir", help="Asset cache; defaults to XDG_CACHE_HOME/strokefill3d"
    )
    parser.add_argument(
        "--brain-mask",
        help="Reviewed native mask; skips skull-stripping model and download",
    )
    parser.add_argument(
        "--synthstrip-weights", help="Local SynthStrip v1 weights for offline use"
    )
    parser.add_argument("--brain-device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument(
        "--registration",
        choices=["affine"],
        default="affine",
        help="Global similarity followed by affine",
    )
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("threads must be positive")
    if bool(args.template) != bool(args.template_mask):
        parser.error("Supply both --template and --template-mask, or neither")
    sitk = sitk_module()
    old_threads = sitk.ProcessObject.GetGlobalDefaultNumberOfThreads()
    try:
        sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(args.threads)
        prepare(args)
    finally:
        sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(old_threads)


if __name__ == "__main__":
    main()
