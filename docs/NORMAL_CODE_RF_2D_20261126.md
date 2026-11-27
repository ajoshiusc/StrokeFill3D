# Normal-code rectified flow: 2D follow-up

**Implementation update:** the selected approach now has a [separate 3D 96³ workflow](NORMAL_PRIOR_4090.md), with a new 3D codec and execution tests. The historical results and limitations below describe the 2D experiment; they do not establish trained 3D quality.

**Result: decoded MRI structure improved materially. The requested oracle-quality recovery and a validated 3D 96³ workflow are not established. No new long 4090 run was started.**

## What changed

The previous template-dependent RF had low interpolation losses but weak generated anatomy. This experiment trains a new MONAI RF prior from independent Gaussian noise to complete nonlesional MRI codes, conditioned only on MNI position. During inpainting, the decoder and RF weights are frozen. Only the initial noise is optimized through a 16-step RF trajectory to match the **visible MRI**. The unknown intensities never enter that optimization.

The setup uses the same project-trained, frozen 2D MRI autoencoder; a training-derived seven-component code basis; a 2.72-million-parameter MONAI U-Net with attention at all three code scales; and 20,000 RF updates. The target bank contains 41,984 crops from 192 training subjects and 2,304 coronal planes. Source lesions and their exclusion margin were rejected. Low-foreground boundary patches were added after identifying that the earlier bank did not cover the difficult partial-context example.

Source fitting uses observed intensity and known-pair gradient errors, emphasizes a four-pixel observed rim, and applies a small source-trust penalty. This is [D-Flow-style optimization](https://proceedings.mlr.press/v235/ben-hamu24a.html) through the learned RF, not exact posterior sampling. There is no critic, retrieved reference patch, image averaging or output sharpening.

## Matched-input results

Means across seeds 909 and 910; the same original image and mask arrays are used for all methods. The cortical group has 16 patches from eight held-out development subjects, not 32 independent subjects. These subjects were excluded from training; they have been inspected during development. The reserved 196-subject test manifest was not used.

| Cohort | Previous latent RF MAE ↓ | New RF MAE ↓ | Previous SSIM ↑ | New SSIM ↑ |
|---|---:|---:|---:|---:|
| Original four cases | 0.233 | 0.199 | 0.417 | 0.499 |
| 16 fixed cortical patches | 0.271 | 0.233 | 0.301 | 0.410 |

The new fills show more curved gray-matter bands and less uniform blur in several holes. They do not consistently reproduce the original fold placement. The largest partial-context example has an incorrect boundary/shape, and one seed introduces a gray island in white matter. Both seeds and every case are retained. Two-start selection using only visible-data energy was audited; it did not improve both cohorts consistently, so it is not required by the selected simple workflow.

## Important negative finding: literal latent proximity did not improve

The AE oracle encodes the **complete original normal image**, including the hidden tissue, and therefore preserves its true folds. It remains around 0.96 SSIM. It is not an inpainting sample and is excluded from selection.

An audit transformed old and new generated states into the same training-derived seven-dimensional code basis. **The new method did not reduce raw code-space MSE to the oracle code**, despite better decoded-image MAE and SSIM. Thus the claim supported here is better decoded MRI structure, not literal recovery of the oracle latent code.

| Cohort | Old full-code MSE | New full-code MSE | Old masked-code MSE | New masked-code MSE |
|---|---:|---:|---:|---:|
| Original four | 0.308 | 0.481 | 0.797 | 0.977 |
| Cortical set | 0.201 | 0.368 | 0.728 | 0.785 |

The masked-code measure uses a spatially pooled mask; code features are not strictly local. Code distance is not a validated measure of anatomical realism, but its failure must not be hidden. Further work is needed before claiming the full requested latent-fidelity goal. Simply projecting the older RF output into the same linear code basis barely changed its image scores, so linear projection alone does not explain the improvement.

## Verification and scope

- Six regression tests passed: straight-flow integration, outside-mask identity, hidden-intensity rejection, MNI coordinate mapping, cancellation of AE round-trip bias, and the source-fitting measurement gradient.
- The portable trainer passed a real forward/backward update, generated validation and atomic checkpoint save. Data preparation ran on a real training subject, and the portable inference CLI ran with the actual trained checkpoint. Smoke outputs are separate from quality results.
- All evaluated predictions were finite and preserved known pixels exactly. Visible-fitting errors were measured on the raw decoder output before compositing; they were not made artificially zero by copying known pixels.
- Training completed at step 20,000. A termination during an earlier save was recovered from a validated checkpoint; saves are now atomic and training completed in a detached tmux session. Torch RNG states are saved in current checkpoints.
- Images now display superior upward. They use the same image arrays as previous comparisons; the vertical rendering correction affects every column equally and does not change the metrics.
- The combined setup changes the prior, training budget, coverage and inference solver. This is not an isolated causal ablation of one component.

## Code, model and results

The portable prototype is in `experiments/normal_code_rf_2d/`. It explicitly supports **2D 64×64** input. The 96³ model is a different architecture and is not replaced or qualified by this test. A separately trained 3D codec must first preserve fine cortical detail, followed by a 3D RF prior and generated validation. The 2D checkpoint cannot be used in a 96³ run.

Local model: `step_020000.pt`; codec: `codec.pt`. Exact hashes are in `summary.json`. MRI-bearing caches, examples and weights remain outside Git.

Comparisons:

- `same_cases_seed909.png` and `same_cases_seed910.png`: all four original examples, including the failure.
- `cortical_01_04_seed909.png` through `cortical_13_16_seed909.png`: all fixed cortical cases; seed 910 counterparts are also included.
- `evaluation/`: raw predictions, generated codes, optimized sources, objective histories and per-case metrics.

**The sensible next step is to review these improved but imperfect 2D results and validate the 3D representation; this report is not a recommendation to launch another long 96³ run yet.**

An optional experiment archive may contain the paired codec, step-20,000 weights, code bank and fixed cases under `$STROKEFILL_DATA_ROOT/experiments/`. These artifacts are separate from all 96³ training runs and are not distributed with this repository.
