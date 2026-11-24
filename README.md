# StrokeFill3D

MONAI research project for **mask-conditioned 3D T1-weighted stroke-lesion inpainting**. The primary model is conditional rectified flow with a 3D MONAI U-Net backbone. A deterministic SwinUNETR baseline is retained in `src/strokefill3d/swin_baseline.py`.

> Research software only. It is not validated for clinical decisions, diagnosis, treatment, or patient care. Outputs are model-generated estimates, not recovered ground truth.

## Why rectified flow

The model learns a velocity field along a straight interpolation from Gaussian noise to a healthy T1 target. Training samples a random point on that path and regresses the target velocity. At inference, random noise is placed inside the lesion mask, the observed T1 is clamped outside the mask, and the learned field is integrated over multiple steps. Different noise seeds can produce different plausible fills. This is flow matching / rectified flow, distinct from FastSurfer-LIT's DDPM reverse-noising procedure. MONAI provides an `RFlowScheduler`; this project uses the same straight-path velocity objective with an explicit masked-context update loop. [MONAI RFlowScheduler documentation](https://monai.readthedocs.io/en/1.5.2/networks.html#rflowscheduler) [Original rectified-flow paper](https://arxiv.org/abs/2209.03003)

Rectified flow is a different generation method, not a guarantee of superior anatomical accuracy. Plausible detail depends on training data and validation. Sample variation is intended; inspect multiple seeds where a single reconstruction would otherwise imply certainty.

## Training design

Training starts from authorized healthy, defaced T1 scans. Each sample gets an irregular synthetic mask. The model receives (1) a partly noise-corrupted target inside the mask, (2) the masked T1 context, and (3) the binary mask. It learns to move the noisy masked region toward the original healthy target. Loss is computed inside the mask. This provides known targets for synthetic inpainting validation.

Synthetic shapes are an initial mask distribution. For stronger stroke realism, use authorized, registered lesion masks to calibrate or replace the synthetic mask sampler. ATLAS lesion scans and masks alone do **not** provide lesion-free paired targets for true inpainting accuracy. Evaluate on held-out healthy scans with synthetic masks, and separately on ethically approved paired longitudinal stroke scans when available.

## Data

The ATLAS stroke dataset can provide realistic lesion-mask shapes, but current releases have access/use requirements. ATLAS v2 access is through ICPSR with data-use requirements; ATLAS v3 requires accepting terms and completing its access process. Follow the dataset owner workflow; this repo includes no patient data. See `docs/DATA.md` and stage authorized scans under `/deneb_disk/data/stroke_inpainting/`.

Create a CSV with a `healthy_t1` column and absolute paths to authorized scans. Split by subject and preferably site before training; never split patches from one scan across train/validation.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

Install CUDA-enabled PyTorch compatible with the target machine before installing this package. Confirm CUDA visibility on the host. Defaults use 48³ patches, batch size 1, mixed precision, and a compact three-level U-Net for the 2060; monitor memory and reduce `patch_size`/channels if needed. On a 4090, increase patch size and channels after a memory check.

## Train

```bash
strokefill-train --config configs/train.yaml \
  --manifest /deneb_disk/data/stroke_inpainting/train_manifest.csv \
  --output /deneb_disk/data/stroke_inpainting/checkpoints
```

## Inference

Image and mask must share the same source voxel grid. The CLI verifies their alignment, then reorients both to RAS and resamples to 1 mm. The output is in that grid. Pixels outside the lesion mask retain the normalized input values. Omit `--seed` for a fresh stochastic completion; pass a fixed seed to reproduce one sample.

```bash
strokefill-infer --image case_T1w.nii.gz --mask case_lesion.nii.gz \
  --checkpoint checkpoints/last.pt --output case_T1w_inpainted.nii.gz \
  --steps 24 --seed 1234
```

## Limitations and validation checklist

- Synthetic masks do not represent every real stroke shape. Validate across lesion sizes, locations, and sites; use authorized real lesion masks to improve the mask sampler.
- A realistic-looking sample can still be anatomically wrong. Report masked MAE/SSIM, boundary error, and outside-mask identity error; assess multiple generated samples and site-held-out cases.
- T1 intensities are percentile-normalized and not calibrated across scanners.
- The model is trained from scratch; dataset size and diversity matter. Do not infer clinical performance from synthetic validation.
- Compare against the retained SwinUNETR baseline and established methods using subject-level splits.

## License

Code: Apache-2.0. Dataset licenses/terms remain separate and govern the data.
