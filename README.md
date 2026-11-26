# StrokeFill3D

MONAI research project for **mask-conditioned 3D T1-weighted stroke-lesion inpainting**. The primary model is conditional rectified flow with a 3D MONAI U-Net backbone. A deterministic SwinUNETR baseline is retained in `src/strokefill3d/swin_baseline.py`.

> Research software only. It is not validated for clinical decisions, diagnosis, treatment, or patient care. Outputs are model-generated estimates, not recovered ground truth.

## Anatomical training on RTX 4090 (current)

Use [the anatomical 4090 workflow](docs/ANATOMICAL_4090.md) for the new 96³ model with observable anatomical cues, EMA, partial-mask supervision and fixed held-out reconstruction validation. Start a **new run** with `configs/atlas3_anatomical_4090.yaml`; old cortical checkpoints have a different architecture. The earlier profiles below are retained for legacy runs.

## Why rectified flow

The model learns a velocity field along a straight interpolation from Gaussian noise to a healthy T1 target. Training samples a random point on that path and regresses the target velocity. At inference, random noise is placed inside the lesion mask, the observed T1 is clamped outside the mask, and the learned field is integrated over multiple steps. Different noise seeds can produce different plausible fills. This is flow matching / rectified flow, distinct from FastSurfer-LIT's DDPM reverse-noising procedure. MONAI provides an `RFlowScheduler`; this project uses the same straight-path velocity objective with an explicit masked-context update loop. [MONAI RFlowScheduler documentation](https://monai.readthedocs.io/en/1.5.2/networks.html#rflowscheduler) [Original rectified-flow paper](https://arxiv.org/abs/2209.03003)

Rectified flow is a different generation method, not a guarantee of superior anatomical accuracy. Plausible detail depends on training data and validation. Sample variation is intended; inspect multiple seeds where a single reconstruction would otherwise imply certainty.

## Training design

Training can use authorized healthy T1 scans or ATLAS3 scans with their paired lesion masks. For ATLAS3, the loader rejects target patches intersecting the known lesion or a conservative surrounding margin, so the target patch contains nonlesional tissue. Donor masks from training sites are then transferred into those target patches. The model receives (1) a noise-corrupted patch inside the donor mask, (2) the masked T1 context, and (3) the donor mask; it learns to move the noisy region toward the known nonlesional target. Loss is computed inside the donor mask.

The ATLAS3 loader preserves real lesion-mask morphology, but randomly flips, rotates, and places masks rather than registering each donor lesion anatomically to the target brain. It requires at least 80% of each mask to overlap T1 foreground and keeps the mask bounding box at least 4 mm from the patch boundary; samples that fail are resampled. The targets are nonlesional patches from stroke scans, not fully healthy control subjects. Validate with site-held-out data and, separately, ethically approved paired longitudinal stroke scans when available.

## Data

The ATLAS stroke dataset can provide realistic lesion-mask shapes, but current releases have access/use requirements. ATLAS v2 access is through ICPSR with data-use requirements; ATLAS v3 requires accepting terms and completing its access process. Follow the dataset owner workflow; this repo includes no patient data. See `docs/DATA.md` and stage authorized scans under `/deneb_disk/data/stroke_inpainting/`.

Build site-held-out ATLAS3 manifests and a training-only donor-mask list:

```bash
python scripts/build_atlas3_training_manifests.py
```

This writes `atlas3_train.csv`, `atlas3_holdout.csv`, and `atlas3_masks_train.csv` under `/deneb_disk/data/stroke_inpainting/manifests/`. The trainer accepts the training manifest as `--manifest` and the donor mask file as `--mask-manifest`. The new anatomical workflow partitions holdout sites into validation and reserved test sets and evaluates fixed reconstructions.

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

Confirm `torch.cuda.is_available()` before training. `configs/atlas3_2060.yaml` is the tested 64³, batch-size-1 profile used on an RTX 2060. It also runs on an RTX 4090 as a conservative starting point. The loader bounds its per-worker volume cache to limit host RAM. On a 4090, increase `patch_size` or batch size only after checking peak memory.

## Train

```bash
python scripts/build_atlas3_training_manifests.py

strokefill-train --config configs/atlas3_2060.yaml \
  --manifest /deneb_disk/data/stroke_inpainting/manifests/atlas3_train.csv \
  --mask-manifest /deneb_disk/data/stroke_inpainting/manifests/atlas3_masks_train.csv \
  --output /deneb_disk/data/stroke_inpainting/runs/atlas3
```

The trainer saves `last.pt` every 250 optimizer steps. Resume a run with:

```bash
strokefill-train --config configs/atlas3_2060.yaml \
  --manifest /deneb_disk/data/stroke_inpainting/manifests/atlas3_train.csv \
  --mask-manifest /deneb_disk/data/stroke_inpainting/manifests/atlas3_masks_train.csv \
  --output /deneb_disk/data/stroke_inpainting/runs/atlas3 \
  --resume /deneb_disk/data/stroke_inpainting/runs/atlas3/last.pt
```

## Cortical recovery fine-tune (RTX 4090)

The first 4090 run completed its configured 20 epochs, but the cortical examples remained mottled. `configs/atlas3_cortical_4090.yaml` continues from that checkpoint with 96³ context, two target patches per scan per epoch, frequent patches centered near the cortical surface, donor masks anchored to cortical/sulcal boundaries, and random 0–2 voxel mask dilation during training. This exposes the model to the mask geometry used at inference and gives it many more examples of intact gyral and sulcal patterns. The 96³ window processes 1.73× as many voxels as 80³; verify peak VRAM on the 4090 before starting the full run.

Run a one-update preflight on the 4090 first. It loads the checkpoint, samples the configured cortical data, runs forward/backward plus one optimizer update, reports peak allocated VRAM, and exits without saving a checkpoint:

```bash
cd /home/ajoshi/Projects/StrokeFill3D
source .venv/bin/activate
PYTHONUNBUFFERED=1 PYTHONPATH=src python -m strokefill3d.train \
  --config configs/atlas3_cortical_4090.yaml \
  --manifest /deneb_disk/data/stroke_inpainting/manifests/atlas3_train.csv \
  --mask-manifest /deneb_disk/data/stroke_inpainting/manifests/atlas3_masks_train.csv \
  --output /tmp/atlas3_cortical_preflight \
  --resume /deneb_disk/data/stroke_inpainting/runs/atlas3_rf_4090_clean/last.pt \
  --preflight
```

If the preflight fits, run the separate fine-tune below. It reads the completed checkpoint and writes to a new run directory:

```bash
cd /home/ajoshi/Projects/StrokeFill3D
source .venv/bin/activate
mkdir -p /deneb_disk/data/stroke_inpainting/runs/atlas3_rf_cortical_4090_20261124/logs
set -o pipefail
PYTHONUNBUFFERED=1 PYTHONPATH=src python -m strokefill3d.train \
  --config configs/atlas3_cortical_4090.yaml \
  --manifest /deneb_disk/data/stroke_inpainting/manifests/atlas3_train.csv \
  --mask-manifest /deneb_disk/data/stroke_inpainting/manifests/atlas3_masks_train.csv \
  --output /deneb_disk/data/stroke_inpainting/runs/atlas3_rf_cortical_4090_20261124 \
  --resume /deneb_disk/data/stroke_inpainting/runs/atlas3_rf_4090_clean/last.pt \
  2>&1 | tee /deneb_disk/data/stroke_inpainting/runs/atlas3_rf_cortical_4090_20261124/logs/train.log
```

After training, preview the held-out cortical case R018/s001 with a two-voxel mask expansion:

```bash
mkdir -p /deneb_disk/data/stroke_inpainting/evaluation/atlas3_cortical_4090_20261124
strokefill-infer \
  --image /deneb_disk/ATLAS3/ATLAS3_Training_Preprocessed/R018/sub-r018s001/ses-1/anat/sub-r018s001_ses-1_space-MNI152NLin2009aSym_desc-brainnorm_T1w.nii.gz \
  --mask /deneb_disk/ATLAS3/ATLAS3_Training_Preprocessed/R018/sub-r018s001/ses-1/anat/sub-r018s001_ses-1_space-MNI152NLin2009aSym_label-lesion_desc-T1lesion_mask.nii.gz \
  --checkpoint /deneb_disk/data/stroke_inpainting/runs/atlas3_rf_cortical_4090_20261124/last.pt \
  --output /deneb_disk/data/stroke_inpainting/evaluation/atlas3_cortical_4090_20261124/R018s001_inpainted.nii.gz \
  --mask-dilation-voxels 2 --steps 48 --seed 2026
```

Inference now zeros the context channel under the mask, as training does, and defaults its window size from the checkpoint's recorded training patch size. The output path above is an evaluation artifact; it does not share the training run directory.

## Cortical recovery fine-tune (RTX 2060)

For a local RTX 2060, use `configs/atlas3_cortical_2060.yaml`. It keeps the known-fit 64³ patch and batch size 1 while enabling cortical patch/mask sampling and 0–2 voxel mask dilation. It resumes the local 2060 checkpoint into a separate run directory:

```bash
cd /home/ajoshi/Projects/StrokeFill3D
source .venv/bin/activate
mkdir -p /deneb_disk/data/stroke_inpainting/runs/atlas3_rf_cortical_2060_20261124/logs
set -o pipefail
PYTHONUNBUFFERED=1 PYTHONPATH=src python -m strokefill3d.train \
  --config configs/atlas3_cortical_2060.yaml \
  --manifest /deneb_disk/data/stroke_inpainting/manifests/atlas3_train.csv \
  --mask-manifest /deneb_disk/data/stroke_inpainting/manifests/atlas3_masks_train.csv \
  --output /deneb_disk/data/stroke_inpainting/runs/atlas3_rf_cortical_2060_20261124 \
  --resume /deneb_disk/data/stroke_inpainting/runs/atlas3_rf_2060_manual/last.pt \
  2>&1 | tee /deneb_disk/data/stroke_inpainting/runs/atlas3_rf_cortical_2060_20261124/logs/train.log
```

## Inference

Image and mask must share the same source voxel grid. The CLI verifies their alignment, then reorients both to RAS and resamples to 1 mm. The output is in that grid. Pixels outside the inpainting mask retain the input values. By default, inference uses 48 flow steps and the patch size recorded in the checkpoint; `--roi-size` and `--steps` can override them. `--mask-dilation-voxels N` expands the supplied mask by N voxels. The default seed is 2026; pass a different `--seed` to explore another stochastic completion.

```bash
strokefill-infer --image case_T1w.nii.gz --mask case_lesion.nii.gz \
  --checkpoint checkpoints/last.pt --output case_T1w_inpainted.nii.gz \
  --steps 48 --seed 1234
```

## Limitations and validation checklist

- ATLAS3 mask-bank sampling preserves real lesion morphology, but randomly places each donor shape rather than anatomically registering it to the target scan. The legacy training commands do not enable validation; the anatomical workflow requires it.
- A realistic-looking sample can still be anatomically wrong. Report masked MAE/SSIM, boundary error, and outside-mask identity error; assess multiple generated samples and site-held-out cases.
- T1 intensities are percentile-normalized and not calibrated across scanners.
- The model is trained from scratch; dataset size and diversity matter. Do not infer clinical performance from synthetic validation.
- Compare against the retained SwinUNETR baseline and established methods using subject-level splits.

## License

Code: Apache-2.0. Dataset licenses/terms remain separate and govern the data.
