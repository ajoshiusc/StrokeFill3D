# External T1 MRI and TBI data

The normal-prior model was trained on skull-stripped T1 scans in **MNI152NLin2009aSym**, with an RAS 1-mm grid. A full-head scan in another MNI template does not match that preprocessing. This workflow prepares native scans for the existing checkpoint; it does not retrain the generator or establish anatomical accuracy on TBI.

## Requirements

Install StrokeFill3D, FreeSurfer's `mri_synthstrip` (including its weights), and ANTs executables on the training or inference computer. Obtain the MNI152NLin2009aSym 1-mm T1 and brain mask, for example through TemplateFlow, under their access terms. Put the tools on `PATH`, or provide `--synthstrip` and `--ants-bin`.

The native T1 and injury labels must already share the same physical grid. For labels drawn on FLAIR, register FLAIR to T1 and transform the labels with nearest-neighbor interpolation first. Copying a header does not align a mask. Singleton fourth dimensions are supported; oblique acquisition geometry is preserved.

## Prepare one case

```bash
python -m strokefill3d.normal_prior.preprocess \
  --image /path/to/native_T1.nii.gz \
  --lesion-labels /path/to/aligned_injury_labels.nii.gz \
  --labels 1 2 \
  --template /path/to/tpl-MNI152NLin2009aSym_res-1_T1w.nii.gz \
  --template-mask /path/to/tpl-MNI152NLin2009aSym_res-1_desc-brain_mask.nii.gz \
  --output-dir /path/to/new_prepared_case
```

The labels default to `1 2`; choose the labels appropriate to the input segmentation. SynthStrip replaces the earlier BET masks that included scalp in the inspected TBI case. A reviewed native brain mask can be supplied with `--brain-mask` to skip extraction. Brain-mask filenames alone do not establish quality.

Preparation runs N4 bias correction using observed brain tissue outside the injury plus a three-voxel margin, followed by lesion-excluded rigid and affine registration to the supplied template. Affine is the default because it is faster and preserves local injury-related shape changes. Add `--registration syn` only when nonlinear registration is needed; it is substantially slower. The same transforms resample T1, brain mask, and lesion mask; masks use nearest-neighbor interpolation. Source images are preserved. A completed preprocessing directory is not overwritten.

Inspect `brain_mask_native.nii.gz`, `brain_n4_native.nii.gz`, `t1_brain_training.nii.gz`, `brain_mask_training.nii.gz`, and `lesion_training.nii.gz`. The JSON reports brain-mask overlap with the template, lesion voxels excluded by brain extraction, and affine determinant/scales (plus warp Jacobians for optional SyN). Numerical gates complement visual inspection; they do not certify registration. Injury labels outside the brain mask require review, especially near a craniectomy or damaged cortex. The effective inference mask is saved explicitly.

## Inpaint

Check the patch count before running the generator:

```bash
python -m strokefill3d.normal_prior.external_infer \
  --prepared /path/to/new_prepared_case \
  --run /path/to/trained_run \
  --output /path/to/results/inpainted_T1.nii.gz \
  --plan-only
```

Remove `--plan-only` to run inference. CUDA, `best.pt`, 16 flow steps, 120 source-fitting updates, seed 909, and two-voxel mask dilation are defaults. `--device cpu` is supported but slow.

Adaptive 96³ windows follow the lesion. Each writes a disjoint interior region with at least 12 voxels to the patch edge, rather than fitting a fixed full-volume grid. All injury labels within each window are hidden together; other generated patches are never used as known anatomy. No anatomical predictions are averaged across windows. Tile ownership boundaries can still produce seams and need visual review.

The input is skull stripped and intensity-scaled with the model's original observed-voxel percentiles. The inverse is a linear scale transformation. The earlier experimental histogram map is not used: clipped quantiles made its inverse ambiguous and exaggerated bright intensities.

Outputs remain on the training template grid in N4-corrected source intensity units. The `.mask.nii.gz` file contains the effective dilated mask restricted to the extracted brain. Outside this mask, output voxels equal the prepared T1 exactly. This preservation does not apply to the original native scan, which has undergone bias correction, registration, and skull stripping.

Tile results are cached beside the output under `.tiles/`. Repeating the identical inference command resumes completed tiles. Input, checkpoint, and codec fingerprints prevent reuse with changed data or settings; use a new output path for comparisons.

Returning a result to native space requires the inverse affine; the optional SyN workflow also requires its saved inverse warp; masks need nearest-neighbor interpolation. The lesion mask should remain available for downstream analyses.

## References

- [SynthStrip usage and citation](https://surfer.nmr.mgh.harvard.edu/docs/synthstrip/)
- [ANTs registration, lesion masks, and preprocessing](https://github.com/ANTsX/ANTs/wiki/Anatomy-of-an-antsRegistration-call)
