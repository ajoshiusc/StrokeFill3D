# Train the mixed-dataset conditional RF model

This is the 3D extension of the successful mixed-dataset **2D conditional
image-space rectified-flow** experiment. It trains directly on complete
normal-appearing 96³ T1 patches. Real injury-mask shapes hide cortical regions;
the RF learns to replace noise with the hidden normal tissue, conditioned on
visible MRI, mask, and MNI coordinates. This is still rectified flow. It has
no separate autoencoder stage and starts with fresh 3D weights. Previous latent
prior checkpoints cannot be resumed in this trainer.

The 2D experiment showed modest improvements across EPISURG, EPIBIOS, and ARC.
This supports a 3D experiment, but does not establish 3D quality. Monitor the
fixed validation images, particularly sulci and gyri in external datasets.

## Start on the training computer

From the existing checkout, with the existing Python environment activated:

```bash
git pull --ff-only
python -m pip install -e . --no-deps
```

No new training dependencies are needed. Full-cohort preparation writes the shared dataset to
`/deneb_disk/data/stroke_inpainting/mixed_rf_v2`. If the disk is mounted elsewhere,
prepare a new cohort at that mount using `STROKEFILL_MIXED_DATA`. CSV paths and
completion hashes are frozen, so editing published CSV files is refused.

First run the full 96³ functional check in its own directory:

```bash
bash scripts/train_mixed_rf_4090.sh /tmp/strokefill_mixed_preflight --preflight
```

Use a new preflight directory if that path already exists. This performs one
training update, generation validation, and checkpoint saving. Its images use
untrained weights. Never use its weights as a trained result.

Start the long run in a persistent terminal session (for example, an existing
`tmux` session). A new directory containing the host and timestamp keeps models separate
from other computers and all previous runs:

```bash
RUN="/deneb_disk/strokefill-runs/mixed_rf96_full_$(hostname)_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN"
printf '%s\n' "$RUN"
bash scripts/train_mixed_rf_4090.sh "$RUN" 2>&1 | tee "$RUN/train.log"
```

Keep the printed path. To resume an interrupted run, use that same path:

```bash
bash scripts/train_mixed_rf_4090.sh "$RUN" --resume 2>&1 | tee -a "$RUN/train.log"
```

In a new shell set `RUN` to the printed directory first. Resume restores model,
EMA, optimizer, mixed-precision scaler, step, and PyTorch RNG state. Configuration
or manifest changes require a fresh run directory. A lock prevents two trainers
from writing the same directory. Preflight runs are marked functional-only and
cannot be resumed as production runs.

## Training and monitoring

The default is 120,000 optimizer updates, one 96³ patch per microbatch and four
microbatches per update. It uses activation checkpointing and mixed precision.
Logs print every 20 updates. Every 500 updates `latest.pt` is saved atomically;
every 2,000 updates fixed generated validation is saved, including comparison
images, masked intensity/gradient error, SSIM, and error outside the mask.
`best.pt` is selected using an equal-weight mean across validation domains.
Checkpoints are also retained every 10,000 updates. Training duration depends
on the computer and data-loading speed; use the logged elapsed time after
several hundred updates to estimate the remaining time.

Review `validation/step_*/comparison.png`, `validation_history.jsonl`, and
`status.json`. Targets, masked input, and generation are shown in that order.
Good training loss alone is insufficient. The final test set is separate and
can be evaluated after model selection:

```bash
bash scripts/train_mixed_rf_4090.sh "$RUN" --evaluate-test
```

The full cohort has separate participants for training, validation, and test in
all four datasets. External splits use a reproducible 80/10/10 division; repeated
scans from a person stay together and one scan is selected per person. The
original ATLAS split is retained. These are development holdouts, not a claim of
independent clinical validation.

## Prepare the full cohort

The earlier `mixed_rf_v1` was a pilot and is refused by the long-run launcher.
The default is now `mixed_rf_v2`. Preparation discovers all labeled participants,
with no subject cap. Scans without an injury label or a paired anatomical scan
are listed in `inventory.json`; registration and target-sampling failures are
recorded separately. Missing labels are never interpreted as normal brain.

```bash
python scripts/prepare_mixed_cohort.py \
  --output /deneb_disk/data/stroke_inpainting/mixed_rf_v2 \
  --synthstrip-weights /path/to/synthstrip.1.pt \
  --template /path/to/MNI_T1.nii.gz \
  --template-mask /path/to/MNI_brain_mask.nii.gz
python scripts/audit_mixed_cohort.py \
  --data /deneb_disk/data/stroke_inpainting/mixed_rf_v2
```

This uses the existing Python preprocessing dependencies. T2/FLAIR injury masks
are registered to the same-session T1 using mutual information; T1 images then
undergo brain extraction, N4 correction, and affine registration to the exact
training grid. All supplied EPISURG rater masks are combined conservatively.
ARC uses its supplied manual injury labels rather than automatic predictions.
The audit checks four lesion-free 96-cube targets from every included scan and
writes anatomical QC images under `qc/`. Review these images for registration
and mask errors; automatic checks cannot prove that all injury is annotated.

`dataset.json` contains final per-domain split counts, exclusions, completion
status, and manifest hashes. The launcher requires the completed audit and
checks hashes before starting. Preparation is resumable at completed subjects;
published cohorts and existing checkpoints require a new directory for changes.
Start a fresh run with the new cohort, using the command above. Do not resume a
pilot checkpoint with changed manifests.

## Data rules

- Half the requested patches come from ATLAS. The other half is shared across
  EPISURG, EPIBIOS, and ARC. Each external cohort uses one scan per person.
- The **entire target patch** must avoid every labeled lesion and a 5-mm 3D
  exclusion margin. This is stricter than masking the lesion out of the loss.
- Brain-only intensity scaling uses p1/p99 of observed, lesion-excluded voxels.
  Preprocessing and inference use the same scaling and grid. ATLAS foreground
  comes from its provided skull-stripped T1; external scans have brain masks.
- Training masks come only from training participants' real binary injury
  contours, placed over normal-appearing cortical regions. No ellipse fallback
  is used. Mild gamma/gain augmentation is retained. The tested combined
  histogram/bias-field augmentation reduced the 2D gain, so it is excluded.
- Supplied injury annotations may miss abnormal tissue. Review the QC overlays;
  the exclusion margin cannot compensate for an entirely missing lesion.
- Data, manifests and checkpoints stay outside Git. The staging script is
  `scripts/prepare_mixed_cohort.py`; it never uses already inpainted MRI as targets.

## Inpaint with the new model

Prepare external T1 scans using the existing Python preprocessing command,
then use the new model's inference entry point:

```bash
strokefill-prepare --image T1.nii.gz --lesion-labels lesion_mask.nii.gz \
  --labels 1 --output-dir prepared_case
python -m strokefill3d.mixed_rf.infer --prepared prepared_case \
  --run "$RUN" --output inpainted_T1.nii.gz
```

Inference uses Euler20 as in the 2D winner, two-voxel mask dilation, direct MRI
conditioning, and MNI location. It handles large masks with owned tiles and
preserves all voxels outside the effective mask. NIfTI output is in template
space. Existing `strokefill-inpaint` continues to load the older latent-prior
model and is not the entry point for this new checkpoint format.
