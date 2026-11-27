# Train the 3D model with an NVIDIA RTX 4090 GPU

An RTX 4090 is a high-memory NVIDIA graphics card (GPU) used here to train the 3D model. This guide starts a **new 3D autoencoder and 3D rectified-flow model from scratch**. The model processes 96×96×96 MRI patches. A short preflight checks that the software and GPU run correctly; it does not train a useful model. The full training job takes much longer and saves checkpoints as it runs. The old 2D and earlier mask-conditioned checkpoints cannot be reused.

## Method and defaults

- MRI patches are **96×96×96 at 1 mm**. The MONAI AutoencoderKL downsamples by only 2 per axis to preserve cortical detail: **48×48×48 codes**, initially eight channels. A training-only standardized PCA removes nearly degenerate code directions; the retained channel count is recorded in `codec.pt`.
- The AE learns full normal tissue reconstruction with intensity, local SSIM and gradient losses, plus mild latent perturbations. Its full-target reconstruction is an **AE oracle diagnostic, not inpainting**. The selected EMA checkpoint must achieve mean masked and foreground SSIM ≥0.93 and worst-case masked SSIM ≥0.85 on four fixed validation cases. The calibrated codec must pass again. Otherwise the process stops before building the RF code bank.
- The RF learns a velocity from independent Gaussian noise to **complete nonlesional codes**, conditioned on MNI world position. No lesion image, copied template, regression completion head, critic or lesion-like target enters this RF objective. The backbone is MONAI DiffusionModelUNet used as a velocity network, not a DDPM sampler.
- Inpainting freezes both networks and fits only the initial Gaussian source through the RF and decoder. The loss sees visible MRI and gradients between known voxel pairs, emphasizes a four-voxel observed rim, and includes a small source-trust penalty. This is [D-Flow-style source optimization](https://proceedings.mlr.press/v235/ben-hamu24a.html), not exact posterior sampling. The final output copies observed voxels exactly.
- Default budgets: AE 20,000 updates; RF 120,000 updates, batch 2 with accumulation 2 (effective batch 4). RF source fitting uses 16 Euler steps and 120 Adam iterations. These are starting budgets, not a promise of convergence or improved anatomy.

## Set up, check the GPU, and start training

Use your existing remote checkout and CUDA environment. Do not start this alongside another GPU trainer.

```bash
cd /home/ajoshi/Projects/StrokeFill3D
git pull
source .venv/bin/activate
python -m pip install -e .

# Actual 96³ forward/backward, checkpoint reload, source gradients and NIfTI IO.
bash scripts/train_normal_prior_4090.sh /tmp/normal_prior_4090_preflight --preflight
```

The preflight uses real training/validation scans with **untrained test weights**, writes under `preflight/`, and exits. Those weights deliberately fail the production codec gate. A preflight preview is not a quality result.

Start a persistent terminal, then a new run:

```bash
tmux new -s normal_prior_4090
cd /home/ajoshi/Projects/StrokeFill3D
source .venv/bin/activate
RUN="/deneb_disk/data/stroke_inpainting/runs/normal_prior_4090_$(hostname -s)_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN/logs"
printf '%s\n' "$RUN" > "$RUN/logs/run_directory.txt"
set -o pipefail
bash scripts/train_normal_prior_4090.sh "$RUN" 2>&1 | tee "$RUN/logs/train.log"
```

Detach with Ctrl-B, then D. Reattach with `tmux attach -t normal_prior_4090`. The pipeline proceeds **AE → capacity gate → frozen/calibrated codec → code cache → RF → full generated validation**. It starts from scratch; do not pass an old checkpoint.

To continue an interrupted run, set `RUN` to its exact existing directory:

```bash
bash scripts/train_normal_prior_4090.sh "$RUN" --resume 2>&1 | tee -a "$RUN/logs/train.log"
```

The new `--resume` is a boolean; it restores the matching stage automatically. Saves are atomic every 250 updates, with online and EMA weights, optimizer, AMP scaler, sampling offset and Torch RNG state. There is no automatic background job or monitoring service.

If the AE gate fails, inspect `ae/validation_*/oracle_*.png` and `ae/latest_validation.json`. Extend the AE without editing the locked configuration:

```bash
bash scripts/train_normal_prior_4090.sh "$RUN" --resume --stage ae --max-steps 40000
bash scripts/train_normal_prior_4090.sh "$RUN" --resume
```

The first command has to pass the gate before continuing. Changing the latent representation after `codec.pt` is frozen would invalidate RF training; the trainer refuses that. Architecture/config changes require a new directory. A compatible 3D AE from this new pipeline may be copied to a new run via `--reuse-ae /path/to/ae/best.pt`; patch size, AE architecture and all manifest hashes must match, and the new run still recalibrates and checks the codec. This option cannot load the earlier 2D or pixel-RF models.

## Shared data and isolation

The launcher uses existing authorized ATLAS3 data and manifests at `/deneb_disk/data/stroke_inpainting/manifests/`:

- Training: `atlas3_train.csv`, 1,172 scans / 44 sites.
- Training-only donor shapes: `atlas3_masks_train.csv`. Used to construct synthetic validation holes, not to train the complete-code RF.
- Validation: `anatomical_v1/atlas3_val.csv`, 84 scans / 5 sites.
- Reserved test: `anatomical_v1/atlas3_test.csv`, 196 scans / 6 sites. Only split metadata is checked; its images are not loaded in training or validation.

Targets are **nonlesional regions of stroke scans**, not a separate cohort of healthy controls. A full crop must avoid the paired real lesion and its three-voxel 3D exclusion margin. Sampling favors cerebral surface regions and includes some low-foreground boundary context. Fixed validation holes are at least eight voxels from the patch edge, contain at least 90% MRI foreground, and use cortical/sulcal anchors rather than cerebellar anchors. These geometric checks are not a segmentation-based guarantee that every hole represents a particular cortical structure.

The data root can be set with `STROKEFILL_DATA_ROOT`; absolute paths inside existing CSVs must still resolve. Each run stores its own AE, codec, code cache, RF, validation images and logs. Use a host/time-specific `RUN`; do not share one output directory between machines. A process lock and hashes of configuration, manifests, codec and cache prevent accidental mixing. A new run refuses unrelated existing contents. File locking depends on the shared filesystem supporting POSIX advisory locks.

The cache contains 8,192 training codes in 16-item shards: about 11.8 GiB at seven FP16 channels or 13.5 GiB at eight, plus metadata. Allow at least 25 GiB per run for cache, checkpoints and validation outputs. The cache is built **after** the new AE passes; the old 2D code bank cannot be reused. MRI, caches and model weights are excluded from Git.

## Validation and outputs

AE reconstruction and actual RF inpainting have separate reports. RF checkpoint selection uses actual inpainting at fixed validation coordinates, masks and seed 909, every 2,000 updates: 8 flow steps and 40 source iterations. It uses a combination of masked MAE, masked-center SSIM and gradient MAE. These are development subjects that have been inspected, not an untouched clinical benchmark.

At the end, the selected best checkpoint is evaluated with the full 16-step/120-iteration protocol and seeds 909 and 910. A short selection protocol can rank checkpoints differently from the full protocol, so inspect the final output before trusting the selection. No oracle target enters source fitting or seed selection.

```text
RUN/run.json                          configuration and manifest identity
RUN/validation_cases.pt               fixed clean targets, masks, observed inputs
RUN/ae/{last,best}.pt                  3D AE checkpoint and optimizer
RUN/ae/validation_*/oracle_*.png       full-target AE diagnostics
RUN/codec.pt                          frozen AE and code transforms
RUN/codec_quality.json                post-calibration capacity gate
RUN/codes/                            separate fingerprinted training-code bank
RUN/prior/{last,best}.pt               new RF checkpoint and paired codec hash
RUN/prior/validation_*/                generated quick-validation PNG/NIfTI/metrics
RUN/prior/final_full_validation/       full-protocol results from selected best RF
```

The previews show three orthogonal planes. Each generated case also contains `target.nii.gz`, `masked_input.nii.gz`, `mask.nii.gz`, and `prediction.nii.gz` in normalized intensity units with its MNI patch affine. Histories report visible fitting error **before** outside-mask compositing.

To repeat full validation:

```bash
bash scripts/train_normal_prior_4090.sh "$RUN" --resume --stage evaluate
```

## Real stroke inference

Input T1 and mask must already be registered to MNI in the same axis-aligned RAS 1-mm grid. `--mni` acknowledges that requirement; an affine check cannot establish successful registration. Normalization excludes all missing intensities.

```bash
python -m strokefill3d.normal_prior.infer \
  --run "$RUN" --image /path/to/MNI_T1.nii.gz \
  --mask /path/to/MNI_stroke_mask.nii.gz \
  --output /path/to/inpainted.nii.gz --mni --device cuda \
  --mask-dilation-voxels 2 --flow-steps 16 --iterations 120 --seed 909
```

Default two-voxel unconditional dilation makes that enlarged mask the generated region. Voxels outside it remain exactly equal to the original float32 T1. The output is in the source T1 intensity units. A JSON sidecar records settings, codec identity, checkpoint step and source-fitting histories.

Each connected component must fit inside one trained 96³ window with eight voxels of observed context on every side. Oversized components fail explicitly. This initial port does not average Gaussian tiles or silently resize anatomy; that would reintroduce an unvalidated source of smoothing. Large lesions beyond that limit require a separately tested sampling strategy.

## Local verification (26 November 2026)

The actual 96³ profile, including RF batch 2 on 48³ codes, passed on the RTX 2060: AE forward/backward and optimizer reload, RF forward/backward, source gradients through frozen RF/AE, rejection of an unqualified codec, and full-volume NIfTI IO with exact preservation outside the mask. Peak allocated VRAM was 1.34 GiB for this short test. This is not a full-training memory benchmark or evidence of trained 3D anatomical quality. Run the preflight on the 4090 as well.

All 23 main unit/regression tests and six portable 2D tests passed. The invariant tests cover flow direction, checkpointed source gradients, reduction of raw visible-data error, hidden-intensity isolation, MNI code coordinates, checkpoint/cache mismatches, interrupted saves, run locking, sampling resume, and rejection of oversized components. A separate private tiny integration test exercises every training stage and RF resume; its untrained AE gate is mocked only inside a temporary test directory, and its weights are deleted. The production gate remains enabled.

The motivating [2D results](NORMAL_CODE_RF_2D_20261126.md) improved decoded structure but retained fold-placement errors. Plausible generated folds should not be interpreted as the patient's recovered pre-stroke anatomy.

To repeat the private integration test (requires the authorized ATLAS manifests and CUDA):

```bash
python tests/run_normal_prior_integration.py --result /tmp/normal_prior_integration.json
```
