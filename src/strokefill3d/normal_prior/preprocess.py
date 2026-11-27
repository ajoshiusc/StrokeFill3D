"""Prepare an external T1 and aligned injury labels for the normal-prior model.

Requires SynthStrip (or a reviewed brain mask), ANTs, and the licensed/accessed
MNI152NLin2009aSym 1-mm T1 template and mask. No dataset is downloaded implicitly.
"""

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.io import loadmat
from scipy.ndimage import binary_dilation

from .io import RunLock, atomic_json


def save(data, reference, path, dtype=np.float32):
    header = reference.header.copy()
    header.set_data_dtype(dtype)
    nib.save(
        nib.Nifti1Image(np.asarray(data, dtype=dtype), reference.affine, header), path
    )


def canonical(path):
    n = nib.load(path)
    a = np.asarray(n.dataobj)
    if a.ndim > 3 and all(v == 1 for v in a.shape[3:]):
        n = nib.Nifti1Image(a.reshape(a.shape[:3]), n.affine, n.header)
    if len(n.shape) != 3 or not np.isfinite(np.asarray(n.dataobj)).all():
        raise ValueError(f"Require a finite 3D NIfTI: {path}")
    return nib.as_closest_canonical(n)


def aligned(a, b):
    if a.shape != b.shape or not np.allclose(a.affine, b.affine, atol=1e-4):
        raise ValueError(
            "Input T1 and labels/mask do not share a grid. Supply verified, aligned masks."
        )


def fingerprint(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ["image", "lesion-labels", "template", "template-mask", "output-dir"]:
        p.add_argument("--" + name, required=True)
    p.add_argument("--labels", nargs="+", type=int, default=[1, 2])
    p.add_argument(
        "--brain-mask",
        help="Reviewed mask aligned with native T1; otherwise run SynthStrip",
    )
    p.add_argument("--synthstrip", default="mri_synthstrip")
    p.add_argument(
        "--ants-bin", default="", help="Directory containing ANTs executables"
    )
    p.add_argument(
        "--ants-library-path",
        help="Optional shared-library path for ANTs subprocesses only",
    )
    p.add_argument(
        "--registration",
        choices=["affine", "syn"],
        default="affine",
        help="Rigid plus affine by default; optional slower nonlinear SyN",
    )
    p.add_argument("--threads", type=int, default=4)
    args = p.parse_args()
    if args.threads < 1:
        raise ValueError("threads must be positive")
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    with RunLock(out):
        if (out / "preprocessing.json").exists():
            raise FileExistsError(
                "Use a new output directory; completed preprocessing will not be overwritten"
            )
        env = os.environ.copy()
        env["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = str(args.threads)
        env["ANTS_RANDOM_SEED"] = "2026"
        if args.ants_library_path:
            env["LD_LIBRARY_PATH"] = args.ants_library_path
        commands = []

        def run(command, log):
            commands.append(list(map(str, command)))
            print("RUN", " ".join(map(str, command)), flush=True)
            with open(out / log, "w") as f:
                subprocess.run(
                    list(map(str, command)),
                    check=True,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    env=(
                        {
                            **os.environ,
                            "OMP_NUM_THREADS": str(args.threads),
                            "MKL_NUM_THREADS": str(args.threads),
                        }
                        if str(command[0]) == args.synthstrip
                        else env
                    ),
                )

        def ants(tool):
            return str(Path(args.ants_bin) / tool) if args.ants_bin else tool

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
        save(raw, native, out / "t1_native.nii.gz")
        save(lesion, native, out / "lesion_native.nii.gz", np.uint8)
        if args.brain_mask:
            masknii = canonical(args.brain_mask)
            aligned(native, masknii)
            brain = np.asarray(masknii.dataobj) > 0
            save(brain, native, out / "brain_mask_native.nii.gz", np.uint8)
        else:
            run(
                [
                    args.synthstrip,
                    "-i",
                    out / "t1_native.nii.gz",
                    "-m",
                    out / "brain_mask_native.nii.gz",
                ],
                "01_synthstrip.log",
            )
            masknii = canonical(out / "brain_mask_native.nii.gz")
            aligned(native, masknii)
            brain = np.asarray(masknii.dataobj) > 0
        fitmask = brain & ~binary_dilation(lesion, iterations=3)
        if fitmask.sum() < 10000:
            raise ValueError(
                "Insufficient observed brain for bias estimation and registration"
            )
        save(fitmask, native, out / "observed_brain_native.nii.gz", np.uint8)
        run(
            [
                ants("N4BiasFieldCorrection"),
                "-d",
                "3",
                "-i",
                out / "t1_native.nii.gz",
                "-x",
                out / "observed_brain_native.nii.gz",
                "-s",
                "3",
                "-b",
                "[200]",
                "-c",
                "[50x50x30x20,1e-6]",
                "-o",
                f"[{out}/t1_n4_native.nii.gz,{out}/bias_native.nii.gz]",
            ],
            "02_n4.log",
        )
        corrected = canonical(out / "t1_n4_native.nii.gz")
        save(
            np.where(brain, np.asarray(corrected.dataobj), 0),
            native,
            out / "brain_n4_native.nii.gz",
        )
        fixed = canonical(args.template)
        fixedmasknii = canonical(args.template_mask)
        aligned(fixed, fixedmasknii)
        if not np.allclose(fixed.affine[:3, :3], np.eye(3), atol=1e-3):
            raise ValueError("Template must have an RAS 1-mm grid")
        fixedbrain = np.asarray(fixedmasknii.dataobj) > 0
        save(
            np.where(fixedbrain, np.asarray(fixed.dataobj), 0),
            fixed,
            out / "template_brain.nii.gz",
        )
        save(fixedbrain, fixed, out / "template_mask.nii.gz", np.uint8)
        f = str(out / "template_brain.nii.gz")
        m = str(out / "brain_n4_native.nii.gz")
        masks = f"[{out}/template_mask.nii.gz,{out}/observed_brain_native.nii.gz]"
        command = [
            ants("antsRegistration"),
            "-d",
            "3",
            "--float",
            "1",
            "--verbose",
            "1",
            "-o",
            f"[{out}/native_to_training_,{out}/registration_preview.nii.gz]",
            "--interpolation",
            "Linear",
            "--winsorize-image-intensities",
            "[0.005,0.995]",
            "--use-histogram-matching",
            "0",
            "-r",
            f"[{f},{m},1]",
        ]
        stages = [
            (
                "Rigid[0.1]",
                f"MI[{f},{m},1,32,Regular,0.25]",
                "[1000x500x250x0,1e-6,10]",
                "8x4x2x1",
                "3x2x1x0vox",
            ),
            (
                "Affine[0.1]",
                f"MI[{f},{m},1,32,Regular,0.25]",
                "[1000x500x250x0,1e-6,10]",
                "8x4x2x1",
                "3x2x1x0vox",
            ),
        ]
        if args.registration == "syn":
            stages.append(
                (
                    "SyN[0.1,3,0]",
                    f"CC[{f},{m},1,4]",
                    "[70x50x30x10,1e-6,10]",
                    "8x4x2x1",
                    "3x2x1x0vox",
                )
            )
        for transform, metric, conv, shrink, smooth in stages:
            command += [
                "-t",
                transform,
                "-m",
                metric,
                "-c",
                conv,
                "-f",
                shrink,
                "-s",
                smooth,
                "-x",
                masks,
            ]
        run(command, "03_registration.log")
        affine_path = out / "native_to_training_0GenericAffine.mat"
        affine_data = loadmat(affine_path)
        parameters = [
            np.asarray(v).ravel()
            for k, v in affine_data.items()
            if k.startswith("AffineTransform_")
        ]
        if len(parameters) != 1 or parameters[0].size != 12:
            raise RuntimeError("Unexpected ANTs affine transform format")
        matrix = parameters[0][:9].reshape(3, 3)
        determinant = float(np.linalg.det(matrix))
        singular_values = np.linalg.svd(matrix, compute_uv=False)
        if (
            not np.isfinite(matrix).all()
            or determinant <= 0
            or singular_values.min() < 0.5
            or singular_values.max() > 2
        ):
            raise RuntimeError(
                "Registration affine has reflection or implausible scales"
            )
        transforms = ["-t", str(affine_path)]
        if args.registration == "syn":
            transforms = [
                "-t",
                str(out / "native_to_training_1Warp.nii.gz"),
            ] + transforms
        for source, destination, interpolation in [
            ("t1_n4_native.nii.gz", "t1_training_fullhead.nii.gz", "Linear"),
            (
                "brain_mask_native.nii.gz",
                "brain_mask_training.nii.gz",
                "NearestNeighbor",
            ),
            ("lesion_native.nii.gz", "lesion_training.nii.gz", "NearestNeighbor"),
        ]:
            run(
                [
                    ants("antsApplyTransforms"),
                    "-d",
                    "3",
                    "-i",
                    out / source,
                    "-r",
                    out / "template_brain.nii.gz",
                    "-o",
                    out / destination,
                    "-n",
                    interpolation,
                ]
                + transforms,
                "04_" + destination + ".log",
            )
        image = canonical(out / "t1_training_fullhead.nii.gz")
        a = np.asarray(image.dataobj, dtype=np.float32)
        bm = np.asarray(canonical(out / "brain_mask_training.nii.gz").dataobj) > 0.5
        lm = np.asarray(canonical(out / "lesion_training.nii.gz").dataobj) > 0.5
        stripped = np.where(bm, a, 0)
        save(stripped, image, out / "t1_brain_training.nii.gz")
        # This reports spatial agreement, but does not certify registration quality.
        dice = (
            2 * int((bm & fixedbrain).sum()) / max(1, int(bm.sum() + fixedbrain.sum()))
        )
        nonpositive = 0  # Positive affine determinant was checked above.
        if args.registration == "syn":
            run(
                [
                    ants("CreateJacobianDeterminantImage"),
                    "3",
                    out / "native_to_training_1Warp.nii.gz",
                    out / "warp_jacobian.nii.gz",
                    "0",
                    "0",
                ],
                "05_jacobian.log",
            )
            jac = np.asarray(canonical(out / "warp_jacobian.nii.gz").dataobj)
            nonpositive = int(((jac[bm] <= 0) | ~np.isfinite(jac[bm])).sum())
        observed = bm & ~lm
        metrics = {
            "brain_mask_dice_with_template": dice,
            "brain_volume_ml": float(bm.sum() / 1000),
            "lesion_voxels": int(lm.sum()),
            "lesion_voxels_outside_brain": int((lm & ~bm).sum()),
            "nonpositive_jacobian_voxels_in_brain": nonpositive,
            "affine_matrix_determinant": determinant,
            "affine_matrix_singular_values": singular_values.tolist(),
            "observed_brain_percentiles": np.percentile(
                a[observed], [1, 10, 50, 90, 99]
            ).tolist(),
            "visual_review_required": True,
        }
        manifest = {
            "template_space": "MNI152NLin2009aSym (caller-supplied template)",
            "registration": args.registration,
            "input_files": {
                str(Path(x).resolve()): fingerprint(x)
                for x in [
                    args.image,
                    args.lesion_labels,
                    args.template,
                    args.template_mask,
                ]
                + ([args.brain_mask] if args.brain_mask else [])
            },
            "brain_extraction": "provided_mask" if args.brain_mask else "SynthStrip",
            "bias_correction": "N4; excludes injury plus 3-voxel margin",
            "intensity_mapping": "none; inference applies training p1/p99 scaling to skull-stripped observed voxels",
            "commands": commands,
            "qc": metrics,
        }
        atomic_json(manifest, out / "preprocessing.json")
        print(json.dumps(metrics, indent=2), flush=True)
        if dice < 0.8 or metrics["nonpositive_jacobian_voxels_in_brain"]:
            raise RuntimeError(
                "Registration QC failed; inspect outputs before inference"
            )


if __name__ == "__main__":
    main()
