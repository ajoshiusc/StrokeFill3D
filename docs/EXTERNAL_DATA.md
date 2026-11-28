# Inpainting a T1 scan from another dataset

The external-data workflow runs inside Python. **No ANTs, FreeSurfer, FSL, BrainSuite or container installation is required.** SimpleITK provides N4 bias correction, affine registration and resampling; PyTorch runs the published SynthStrip v1 brain-extraction model.

The software supports research evaluation on external T1 MRI. Current checkpoints still have limitations in cortical detail; simplifying installation does not establish anatomical accuracy on a new dataset.

## Install once

From the repository checkout, activate a Python environment with CUDA-enabled PyTorch 2.6 and run:

```bash
python -m pip install -e '.[preprocess]'
```

The extra installs **SimpleITK** from pip. All other Python requirements are declared by the project. Standard Linux x86-64 wheels provide the compiled libraries; no separate ITK installation is needed. Python 3.10 is tested. An NVIDIA RTX 4090 is recommended for the 96³ generator; preprocessing uses the CPU by default.

The exact MNI152NLin2009aSym 1-mm template, its brain mask and SynthStrip weights download on first use. Downloads are cached and verified against SHA-256 hashes. To download them before processing, including before moving to an offline computer:

```bash
strokefill-download-assets --cache-dir /path/to/asset_cache
```

Without `--cache-dir`, the cache is `$XDG_CACHE_HOME/strokefill3d`, or `~/.cache/strokefill3d`. These assets are separate from the **trained inpainting model**, which is not included in the repository. The model run must contain matching `codec.pt` and `prior/best.pt` files.

## Prepare a case

The input T1 and lesion segmentation must already share the same physical grid. A segmentation drawn on another sequence must first be registered to T1 and resampled with nearest-neighbor interpolation. Copying a header does not align a mask.

For a binary lesion mask:

```bash
strokefill-prepare \
  --image /path/to/T1.nii.gz \
  --lesion-labels /path/to/lesion_mask.nii.gz \
  --labels 1 \
  --output-dir /path/to/prepared_case
```

For a multilabel segmentation, select the labels to fill, for example `--labels 1 2`. That is also the default label selection. A reviewed native brain mask, including one produced by BrainSuite, can be supplied with `--brain-mask`; this skips SynthStrip and its weight download. No BrainSuite installation is needed by the default workflow.

The command performs brain extraction, lesion-excluded N4 correction, global similarity alignment followed by affine registration, and resampling to the training space. Registration is affine only. Brain physical volume initializes global scale so scanner/template size differences do not depend on starting from unit scale. Optimization uses bounded physical steps, injury-excluded masks and multiresolution pyramids. Similarity alignment uses mutual information; the final affine stage uses correlation for T1-to-T1 alignment. There is no nonlinear registration stage.

N4 and intensity normalization for registration exclude injury plus a three-voxel margin. The normal-prior inference code separately scales observed voxels using its training percentiles. No histogram transfer is applied to generated tissue.

For offline use or an existing template installation, supply **both** `--template` and `--template-mask`, plus `--synthstrip-weights` or `--brain-mask`. The template must be the model's MNI152NLin2009aSym T1 with an RAS 1-mm grid. The same options work through `python -m strokefill3d.normal_prior.preprocess`.

## Inspect preparation

Inspect these NIfTI files before generation:

- `brain_mask_native.nii.gz` and `brain_n4_native.nii.gz`;
- `t1_brain_training.nii.gz`, `brain_mask_training.nii.gz` and `lesion_training.nii.gz`.

`preprocessing.json` records the Python backend/version, input hashes, skull-stripping weights, registration diagnostics, affine scales/determinant, template-brain overlap and injury voxels outside brain extraction. Registration overlap below 0.8 or an invalid affine stops the workflow. Numerical checks do not certify alignment, especially near damaged cortex or a craniectomy.

Original inputs are preserved. Completed preparation directories are not overwritten. Use a new directory when changing preprocessing.

## Inpaint

```bash
strokefill-inpaint \
  --prepared /path/to/prepared_case \
  --run /path/to/trained_run \
  --output /path/to/inpainted_T1.nii.gz
```

Add `--plan-only` to inspect the number of windows without running the generator. CUDA, `best.pt`, 16 flow steps, 120 source-fitting updates, seed 909 and two-voxel mask dilation are defaults. `--device cpu` is supported but slow. The module equivalent is `python -m strokefill3d.normal_prior.external_infer`.

Adaptive 96³ windows cover the lesion. Each writes a disjoint interior region with at least 12 voxels to the patch edge. All injury labels within a window are hidden together. No anatomical predictions are averaged across windows. Tile boundaries can still produce seams.

Outputs remain on the training template grid in N4-corrected source intensity units. The `.mask.nii.gz` file contains the effective dilated mask restricted to the extracted brain. Outside that mask, output voxels equal the **prepared T1** exactly. Injury outside the extracted brain requires review. This preservation does not apply to the original native full-head scan.

Tile results are cached beside the output under `.tiles/`. Repeating an identical command resumes completed tiles. Input, checkpoint, codec and inference-code fingerprints prevent accidental reuse after changes. Use a new output path for a new preprocessing result or comparison.

## Transforms and compatibility

Preparation saves standard ITK text transforms, readable using `SimpleITK.ReadTransform`:

- `training_to_native.tfm` maps template LPS physical points to native LPS physical points. It is the pull transform used to resample native MRI onto the template.
- `native_to_training.tfm` is its inverse, used to resample a template-space result onto the saved native grid.

Neither direction needs an ANTs executable. Native export remains an additional resampling step; restoration of the original intensity bias and preservation of native observed tissue must be handled explicitly.

The new preparation backend does not change the trained RF architecture or checkpoint format. Existing prepared directories remain readable by inference. New preprocessing is not numerically identical to the former ANTs pipeline; it must use a fresh preparation directory. Former ANTs executable options and optional SyN registration are no longer supported by preparation.

## Validation and attribution

Tests cover physical RAS/LPS conversion including oblique scans, transform direction and inverse resampling, binary masks, recovery of a known affine, injury-excluded N4 correction, checksummed assets, and execution without subprocesses. Private real-case checks also compare extraction and affine alignment with the former pipeline.

The SynthStrip v1 weights and template files come from their original providers. See [third-party notices](../NOTICE). Relevant references:

- [SynthStrip model, weights and licensing](https://surfer.nmr.mgh.harvard.edu/docs/synthstrip/)
- [SimpleITK](https://simpleitk.org/)
- [TemplateFlow MNI152NLin2009aSym](https://github.com/templateflow/tpl-MNI152NLin2009aSym)
