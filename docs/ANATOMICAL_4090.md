# Anatomical rectified flow on RTX 4090

This is a new model and a new training run. It is intended to address the mottled white matter and weak cortical structure seen in the earlier three-level model. Better output quality remains to be established by validation and inspection. Do not resume an old 3-channel/three-level checkpoint into this model.

## Start on the remote machine

From your existing checkout and environment:

```bash
cd /home/ajoshi/Projects/StrokeFill3D
git pull --ff-only
source .venv/bin/activate
# Reuse the working PyTorch/MONAI environment from the previous run.

# One real training update and a short generated validation reconstruction.
# This reports actual peak VRAM on YOUR 4090 and writes no training checkpoint.
bash scripts/train_anatomical_4090.sh /tmp/strokefill_anatomical_preflight --preflight

# Use a distinct directory for this host and attempt, on the shared data disk.
RUN="/deneb_disk/data/stroke_inpainting/runs/anatomical_4090_$(hostname -s)_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN/logs"
printf '%s\n' "$RUN" > "$RUN/logs/run_directory.txt"
set -o pipefail
bash scripts/train_anatomical_4090.sh "$RUN" 2>&1 | tee "$RUN/logs/train.log"
```

No new runtime dependency is required for this upgrade; the launcher imports the pulled source directly. For a fresh environment, use Python 3.10/3.11 and the installation instructions in README.md. Use tmux/screen for a long training session. Stop an existing trainer separately if it occupies the 4090. This script does not terminate other processes. `STROKEFILL_DATA_ROOT` can change the shared data root; existing CSV image paths must still resolve. Read the generated train log for sampling or CUDA failures before leaving the job unattended.

To resume this new run after interruption, set RUN to its existing directory:

```bash
bash scripts/train_anatomical_4090.sh "$RUN" --resume "$RUN/last.pt" 2>&1 | tee -a "$RUN/logs/train.log"
```

`--resume` restores online weights, EMA, optimizer, scaler, global update count and best score. Mid-epoch resumes replay that epoch's data sampling; exact minibatch continuation is not promised. The global update limit still applies. A completed run requires raising max_steps/epochs in a deliberately new config to extend it.

## What changed

- 96³, four scales [32,64,128,256], two residual blocks, attention at the coarsest scale, memory-efficient attention and activation checkpointing. The local 2060 preflight is a functional check; the remote 4090 preflight is the authoritative check of its installed environment and VRAM.
- Ten input channels: evolving T1, masked context, lesion mask, plus seven anatomical channels: opposite-hemisphere observed intensity and validity, MNI xyz coordinates, and coarse observed intensity and support. The opposite hemisphere is reflected about MNI world x=0, not the array midpoint. No segmentation model or additional dataset is required.
- Hidden synthetic tissue and real source lesions are removed before generating every image-based anatomical cue. Invalid mirror values are zeroed. The coarse cue uses normalized Gaussian convolution of visible tissue; it is not a clean-target conditioning image. Mirror symmetry is approximate and is not a recovered personal healthy anatomy.
- Source target patches must avoid the source lesion plus a three-voxel exclusion margin. Synthetic lesion masks are placed on the full scan before cropping. Thirty-five percent of requested placements touch patch faces, providing supervision at the faces used by tiled inference. Sixty percent request large masks. Failed placements are retried; actual accepted mix is recorded by the validation-case manifest and can be inspected. Cerebral sampling is restricted to MNI z > -10 mm, with at least 90% of full synthetic mask voxels above that plane. Surface/deep regions use distance-to-foreground heuristics, not validated GM/WM labels.
- Flow MSE remains primary. Small late-time clean-endpoint L1 and gradient terms encourage target contrast and continuity, including mask boundaries. Gradient accumulation gives an effective batch of two. EMA, LR warmup and cosine decay are enabled. These auxiliary weights are conservative starting choices, not tuned optimum values.

## Data and validation

The existing training manifest and training-only donor masks are reused. `split_validation_sites.py` divides the existing holdout into a reproducible five-site validation set (84 scans here) and a six-site reserved test set (196 scans here). It refuses to overwrite a different split. The trainer checks subject/site separation and donor membership. Test images are not loaded for optimization or checkpoint selection. Previously inspected real lesions are development examples, not evidence of untouched test performance.

Eight fixed synthetic examples span large cortical, large deep, partial large cortical and small deep masks. The full cached targets/conditions and source metadata are written once per run and reused; changed split/config hashes are rejected. Each evaluation reconstructs them from fixed noise at two seeds, using 48 steps. Evaluation occurs before training, every 2,000 successful optimizer updates, and at epoch/run end. All metrics use EMA when available. Example seeds are scored individually; final images are never averaged across seeds.

`best.pt` minimizes:

```
masked_MAE + 0.25 * (1 - bounding_box_SSIM) + 0.25 * gradient_MAE
```

MAE and gradient error are masked. SSIM is measured on the mask bounding box plus three voxels of context, so it is not a fully masked SSIM. Metrics use normalized [-1,1] intensity, not scanner units. PSNR and a high-pass error in eroded bright-tissue regions are also recorded; the latter is explicitly a white-matter PROXY, not a WM segmentation metric. Those proxy regions are determined from the known target only for scoring, never model conditioning. Eight validation examples are a practical initial panel, not a comprehensive benchmark; increase `validation_cases` in a fresh run for broader evaluation.

Files:

- `validation_history.jsonl`: aggregate reconstruction metrics and selection score.
- `validation/step_*/metrics.json`: individual cases/seeds and aggregate results.
- `validation/step_*/case_*.png`: rows axial/coronal/sagittal; columns masked input/known target/generated output. Shared normalized intensity display.
- `validation/step_*/reconstructions.pt`: CPU tensors of target, prediction and mask for first seed.
- `validation_cases.json` and `validation_cases.pt`: fixed case identities and cached tensor pairs.
- `training_history.jsonl`: flow, endpoint and gradient losses, LR and gradient norm.
- `last.pt`, `best.pt`, `epoch_*.pt`: checkpoints with online and EMA weights. Epoch checkpoints can require tens of GB across a long run; set keep_epoch_checkpoints false to keep last/best only.

The default cap is 120,000 successful optimizer updates or 80 cumulative epochs, whichever comes first. Choose duration by held-out structural quality. No claim is made that the cap guarantees convergence. Keep a final test set separate from tuning these weights.

## Inference from the new checkpoint

Supply the same ATLAS3 preprocessed 1-mm MNI-space image and its aligned lesion mask:

```bash
strokefill-infer --image /path/to/MNI_T1w.nii.gz --mask /path/to/MNI_lesion_mask.nii.gz \
  --checkpoint "$RUN/best.pt" --output "$RUN/evaluation/case_inpainted.nii.gz" \
  --mni --strategy auto --mask-dilation-voxels 2 --steps 48 --seed 2026 --amp
```

EMA weights are used automatically. `--strategy auto` selects a centered 96³ ROI if the full mask fits with at least eight voxels of context on every face, otherwise Gaussian-weighted 50%-overlap tiling. `--strategy gaussian` forces tiling. Observed output intensities are preserved outside the inpainting mask. Output NIfTIs use the RAS/1-mm preprocessing grid and include an adjacent JSON record. Old checkpoints are supported without `--mni` and without anatomical channels; sampling now defaults to centered-or-Gaussian rather than old constant-weight tiling.

Anatomical conditioning assumes actual MNI registration. The code can check voxel geometry, but cannot prove registration from an affine. Do not pass native patient images with `--mni` merely to bypass the check. The code does not perform registration. For generic native-space work, first register the T1 and mask consistently to MNI and validate the alignment.

## Development checks

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

The regression checks cover hidden-target and real-lesion leakage through conditioning, mirror geometry, crop equivalence, Gaussian integration, observed-voxel preservation, split/donor isolation, anatomical backward propagation and reconstruction metrics. The real-data preflight additionally checks the actual 96³ model update and a short validation sample. It is not evidence of trained reconstruction quality.
