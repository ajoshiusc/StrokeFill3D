# StrokeFill3D

MONAI research project for **mask-conditioned inpainting of stroke lesions in T1-weighted brain MRI**. The first model is a 3D SwinUNETR that predicts the missing healthy tissue in one forward pass. It is architecturally distinct from FastSurfer-LIT's DDPM-based lesion inpainting. MONAI 1.6.1 includes recent SwinUNETR updates, including optional flash attention support.

> Research software only. It is not validated for clinical decisions, diagnosis, treatment, or patient care. Outputs are model-generated estimates, not recovered ground truth.

## Training design

Training starts from authorized healthy, defaced T1 scans. At each sample, the loader creates a connected irregular synthetic mask, zeros those voxels, and trains the network to reconstruct the original T1. The loss weights masked-region L1 error and lightly penalizes changes outside the mask. The model gets both masked image and mask as input. For actual lesion scans, inference requires a separately supplied lesion mask. If the acquisition, anatomy, lesion appearance, or preprocessing distribution differs from training, outputs may be inaccurate.

This is a self-supervised synthetic-corruption baseline. Validate on held-out healthy scans with synthetic masks and, separately, on ethically approved paired longitudinal stroke scans when available. ATLAS lesion T1 and lesion masks alone do **not** provide the lesion-free target needed to compute true inpainting accuracy.

## Data

The ATLAS stroke dataset can provide realistic lesion-mask shapes for future mask sampling, but ATLAS v2 is distributed under data-use requirements and ATLAS v3 requires accepting terms and an access request. Complete the dataset owner workflow and record its terms before downloading or redistributing anything. This repo intentionally contains no patient data. See `docs/DATA.md` and stage authorized scans under `/deneb_disk/data/stroke_inpainting/`.

Create a CSV with a `healthy_t1` column. Paths can be absolute. Split subjects/cohorts before training; never split patches from a scan across train and validation.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

Install a CUDA-enabled PyTorch build compatible with the target machine before installing this package. Confirm `torch.cuda.is_available()` and run a one-batch memory check on the target GPU. The current host has no accessible NVIDIA driver; a 4090 is likely appropriate for 96³ patches with batch size 1 and AMP/checkpointing, but adjust patch size/features if memory is insufficient.

## Train

```bash
strokefill-train --config configs/train.yaml \
  --manifest /deneb_disk/data/stroke_inpainting/train_manifest.csv \
  --output /deneb_disk/data/stroke_inpainting/checkpoints
```

## Inference

Image and mask must be in the same voxel grid with matching shape and affine. Use the lesion mask in native/preprocessed space matching the scan; inference preserves input voxels outside the mask.

```bash
strokefill-infer --image case_T1w.nii.gz --mask case_lesion.nii.gz \
  --checkpoint checkpoints/last.pt --output case_T1w_inpainted.nii.gz
```

## Limitations and validation checklist

- Synthetic lesions do not reproduce every T1 stroke appearance. Tune mask distributions to authorized ATLAS masks and evaluate across held-out sites.
- T1 intensities are only percentile-normalized; the model is not yet calibrated across scanners.
- The training loader uses NIfTI voxel orientation/resampling; ensure registered, defaced inputs and common field of view.
- Report masked MAE/SSIM, boundary error, and outside-mask identity error; include lesion size/location strata and site-held-out validation.
- Compare against the unmodified input and established baselines. Do not claim clinical utility from synthetic validation.

## License

Code: Apache-2.0. Dataset licenses/terms remain separate and govern the data.
