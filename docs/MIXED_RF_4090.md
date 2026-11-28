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

No new dependencies are needed. The prepared shared dataset is already at
`/deneb_disk/data/stroke_inpainting/mixed_rf_v1`. If the disk is mounted elsewhere,
set `STROKEFILL_MIXED_DATA` and rewrite the paths in copies of the three CSV files
for that mount. CSV file paths are absolute.

First run the full 96³ functional check in its own directory:

```bash
bash scripts/train_mixed_rf_4090.sh /tmp/strokefill_mixed_preflight --preflight
```

Use a new preflight directory if that path already exists. This performs one
training update, generation validation, and checkpoint saving. Its images use
untrained weights. Never use its weights as a trained result.

Start the long run in a persistent terminal session (for example, an existing
`tmux` session). A new directory on the training computer keeps models separate
from other computers and all previous runs:

```bash
RUN="$HOME/strokefill-runs/mixed_rf96_$(hostname)_$(date +%Y%m%d_%H%M%S)"
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

The current validation includes all four domains. The test set includes ATLAS
and a separate ARC participant. More EPISURG/EPIBIOS participants are needed
for a separate final test of those two domains; the present external sample is
small and has already informed experiment design.

## Data rules

- Half the requested patches come from ATLAS. The other half is shared across
  EPISURG, EPIBIOS, and ARC. Within ARC the staged subset uses one scan per person.
- The **entire target patch** must avoid every labeled lesion and a 5-mm 3D
  exclusion margin. This is stricter than masking the lesion out of the loss.
- Brain-only intensity scaling uses p1/p99 of observed, lesion-excluded voxels.
  Preprocessing and inference use the same scaling and grid. ATLAS foreground
  comes from its provided skull-stripped T1; external scans have brain masks.
- Training masks come only from training participants' real binary injury
  contours, placed over normal-appearing cortical regions. No ellipse fallback
  is used. Mild gamma/gain augmentation is retained. The tested combined
  histogram/bias-field augmentation reduced the 2D gain, so it is excluded.
- The existing ARC derivatives contain automatically predicted stroke masks,
  not verified manual annotations. Missed lesions can still contaminate targets
  despite the margin. Review the staged mask overlays and replace inaccurate
  exclusions before a definitive run. `dataset.json` records this provenance.
- Data, manifests and checkpoints stay outside Git. The staging script is
  `scripts/prepare_mixed_rf.py`; it never uses already inpainted MRI as targets.

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
