# StrokeFill3D

MONAI research project for **mask-conditioned 3D T1-weighted stroke-lesion inpainting**. The primary model is conditional rectified flow with a 3D MONAI U-Net backbone. A deterministic SwinUNETR baseline is retained in `src/strokefill3d/swin_baseline.py`.

> Research software only. It is not validated for clinical decisions, diagnosis, treatment, or patient care. Outputs are model-generated estimates, not recovered ground truth.

## Why rectified flow

The model learns a velocity field along a straight interpolation from Gaussian noise to a healthy T1 target. Training samples a random point on that path and regresses the target velocity. At inference, random noise is placed inside the lesion mask, the observed T1 is clamped outside the mask, and the learned field is integrated over multiple steps. Different noise seeds can produce different plausible fills. This is flow matching / rectified flow, distinct from FastSurfer-LIT's DDPM reverse-noising procedure. MONAI provides an `RFlowScheduler`; this project uses the same straight-path velocity objective with an explicit masked-context update loop. [MONAI RFlowScheduler documentation](https://monai.readthedocs.io/en/1.5.2/networks.html#rflowscheduler) [Original rectified-flow paper](https://arxiv.org/abs/2209.03003)

Rectified flow is a different generation method, not a guarantee of superior anatomical accuracy. Plausible detail depends on training data and validation. Sample variation is intended; inspect multiple seeds where a single reconstruction would otherwise imply certainty.

## Training design

Training can use authorized healthy T1 scans or ATLAS3 scans with their paired lesion masks. For ATLAS3, the loader rejects target patches intersecting the known lesion or a conservative surrounding margin, so the target patch contains nonlesional tissue. Donor masks from training sites are then transferred into those target patches. The model receives (1) a noise-corrupted patch inside the donor mask, (2) the masked T1 context, and (3) the donor mask; it learns to move the noisy region toward the known nonlesional target. Loss is computed inside the donor mask.

The ATLAS3 loader preserves real lesion-mask morphology, but randomly flips, rotates, and places masks rather than registering each donor lesion anatomically to the target brain. The targets are nonlesional patches from stroke scans, not fully healthy control subjects. Validate with site-held-out data and, separately, ethically approved paired longitudinal stroke scans when available.

## Data

The ATLAS stroke dataset can provide realistic lesion-mask shapes, but current releases have access/use requirements. ATLAS v2 access is through ICPSR with data-use requirements; ATLAS v3 requires accepting terms and completing its access process. Follow the dataset owner workflow; this repo includes no patient data. See `docs/DATA.md` and stage authorized scans under `/deneb_disk/data/stroke_inpainting/`.

Build site-held-out ATLAS3 manifests and a training-only donor-mask list:

```bash
python scripts/build_atlas3_training_manifests.py
```

This writes `atlas3_train.csv`, `atlas3_holdout.csv`, and `atlas3_masks_train.csv` under `/deneb_disk/data/stroke_inpainting/manifests/`. The trainer accepts the training manifest as `--manifest` and the donor mask file as `--mask-manifest`. The holdout is reserved for a later evaluation step; the current trainer does not yet run validation.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

The 2060 host has driver 550.163.01 (CUDA 12.4). MONAI 1.5.2 supports PyTorch 2.6 and includes rectified-flow support, so use the matching CUDA 12.4 wheel before installing the project:

```bash
python -m pip install torch==2.6.0 torchvision==0.21.0 \
  --index-url https://download.pytorch.org/whl/cu124
python -m pip install -e .
```

Confirm `torch.cuda.is_available()` before training. Defaults use 64³ patches, batch size 1, two data workers, four sampled patches per scan per epoch, mixed precision, and a compact three-level U-Net for the 2060; monitor memory and reduce `patch_size`/channels if needed. The loader bounds its per-worker volume cache to limit host RAM. On a 4090, increase patch size and channels after a memory check.

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

- ATLAS3 mask-bank sampling preserves real lesion morphology, but randomly places each donor shape rather than anatomically registering it to the target scan. The held-out site manifest is written, but validation is not yet implemented in the trainer.
- A realistic-looking sample can still be anatomically wrong. Report masked MAE/SSIM, boundary error, and outside-mask identity error; assess multiple generated samples and site-held-out cases.
- T1 intensities are percentile-normalized and not calibrated across scanners.
- The model is trained from scratch; dataset size and diversity matter. Do not infer clinical performance from synthetic validation.
- Compare against the retained SwinUNETR baseline and established methods using subject-level splits.

## License

Code: Apache-2.0. Dataset licenses/terms remain separate and govern the data.
