# Experimental 2D normal-code rectified flow

This is the tested 64×64 coronal prototype, **not the 96³ training workflow**. It trains a position-conditioned healthy-anatomy RF prior on complete nonlesional MRI codes, then fits the Gaussian source through the frozen flow to the visible MRI. No hidden target intensities enter inference. The generator remains rectified flow; no critic, reference-patch retrieval or image averaging is used.

The frozen MONAI autoencoder is from this project's own training, not external pretrained weights. The 8-channel code is centered and transformed into seven well-scaled informative directions using training data only. The RF backbone is a MONAI DiffusionModelUNet with attention at all three latent scales, two residual blocks per scale and 2.72 million parameters.

## Observed result

The completed 20,000-update experiment used 192 ATLAS3 training subjects, 2,304 coronal planes and 41,984 encoded nonlesional crops. Across seeds 909 and 910, masked SSIM improved approximately **0.301 → 0.410** on 16 fixed cortical patches from eight held-out subjects, and **0.417 → 0.499** on the earlier four examples. These changes combine a different prior, more training, better coverage and a different inference solver; they are not an isolated architecture ablation.

Curved gray-matter bands are more visible, but some folds remain misplaced, a gray island can appear in white matter, and the largest partial-context example is unreliable. The AE oracle remains much better because it receives the complete original image. A common-basis code audit did not show lower raw latent MSE; the measured improvement is in decoded MRI structure and image fidelity, not literal proximity to the oracle code. **Do not treat these results as validation of a 3D model or launch the old 96³ configuration expecting this method.**

Source fitting follows the idea of [D-Flow](https://proceedings.mlr.press/v235/ben-hamu24a.html), with a visible-MRI intensity/gradient objective, an observed boundary-rim weight and a source-trust penalty. This is regularized optimization, not an exact posterior sampler. Learning the complete normal-anatomy prior is motivated by [healthy-anatomy flow priors](https://arxiv.org/abs/2610.08983); this latent implementation is not a reproduction of that paper.

## Environment and input format

Use the project's PyTorch 2.6 / MONAI 1.5.2 environment. PNG rendering additionally uses Pillow. The included scripts are intentionally separate from the production 3D CLI.

Prepared cases are a `.pt` dictionary containing:

- `image`: float32 `B×1×64×64`, normalized to `[-1,1]` using full-volume percentiles. The CLI zeros masked pixels before generation.
- `mask`: binary `B×1×64×64`, one where tissue is unknown.
- `coords`: `B×3×16×16`, average-pooled MNI xyz coordinates divided by 100; alternatively `anatomy` with xyz in channels `2:5` at full resolution.
- Optional `target`: original normal images, accessed **only after inference** for scoring/display.

The frozen codec and RF checkpoint must come from the same run. Their hashes are recorded by inference. The code explicitly rejects 3D input and non-64×64 images.

## Inference

```bash
python experiments/normal_code_rf_2d/infer_cases.py \
  --run "$RUN" --checkpoint step_020000.pt \
  --cases "$CASES" --output "$RESULT" \
  --method source --steps 16 --iterations 120 \
  --trust 0.001 --boundary-weight 4 --seed 909
```

For a dictionary containing multiple cohorts, add e.g. `--key original_cases`. Repeat with seed 910 to inspect ambiguity. Each result is one generated sample. Two-start selection by visible-data fit was also audited; it did not consistently improve both cohorts, so it is not a required component.

## Train the RF prior from an existing compatible code cache

```bash
python experiments/normal_code_rf_2d/train.py \
  --output "$NEW_RUN" --steps 20000 --batch 48 --compile \
  --bank-cache "$CODES" --frozen-codec "$CODEC" \
  --validation-pairs "$VALIDATION_CASES"
```

Use a new directory for a new experiment. To continue an existing run, pass its `--output` and `--resume "$RUN/last.pt"` with a larger `--steps` limit. Checkpoint writes are atomic, and current checkpoints retain Torch RNG states. Use tmux for sustained runs.

To prepare a new plane bank from an independent **training-only** ATLAS3 manifest:

```bash
python experiments/normal_code_rf_2d/prepare_planes.py \
  --manifest "$TRAIN_MANIFEST" --output "$PLANE_BANK" --subjects 192

python experiments/normal_code_rf_2d/train.py \
  --plane-bank "$PLANE_BANK" --autoencoder-checkpoint "$OWN_2D_AE" \
  --validation-pairs "$VALIDATION_CASES" --output "$NEW_RUN" --compile
```

The preparation script requires the canonical 1-mm ATLAS3 MNI grid and excludes source lesions plus a three-voxel 3D margin. Keep validation subjects and sites outside the training manifest. Boundary patches are included. The new preparation script uses a more conservative 3D exclusion than the historical slice-bank prototype; it is a portable implementation of the recipe, not a promise of identical training streams or weights. A supplied autoencoder checkpoint needs MONAI `config`, `model`, `mean` and `std` fields; the trainer computes the code basis from training data.

## Checks

```bash
python experiments/normal_code_rf_2d/test_core.py
```

Tests cover exact straight-flow integration, preservation of known pixels, rejection of hidden input intensities, coordinate consistency, cancellation of AE round-trip bias and source-fitting measurement gradients. The exported trainer also passed a real forward/backward update, checkpoint save and generated validation on the RTX 2060. The preparation script was exercised on a real training subject.

No MRI, code banks or model weights belong in Git. The measured experiment and remaining failures must be reviewed before extending the method to a separately trained 3D codec and 96³ RF prior.
