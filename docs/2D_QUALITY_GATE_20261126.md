# Rectified-flow 2D quality review — 26 November 2026

**Verdict: quality gate NOT PASSED. No new model is qualified for a long 96³ RTX 4090 run.**

The revised RF improves aggregate error over the original pixel RF, but the images still do not demonstrate convincing recovery of the missing sulcal and gyral pattern. The auxiliary critic does not justify its extra complexity. A favorable autoencoder reconstruction must not be presented as evidence of successful inpainting.

## What “AE oracle” means

AE means autoencoder. The oracle diagnostic encodes the complete original normal patch, including the hidden tissue, and decodes that code. It tests whether the representation can preserve the true anatomy. It is not an inpainting result and is excluded from model selection. Its masked SSIM is approximately 0.96 on the new centered cases. The actual RF model receives masked context and mask-safe observable anatomical features; it must generate the unknown code itself. That generation remains the failing part.

## Controlled evaluation

- Actual ATLAS3 T1 data: nonlesional target patches from stroke scans, excluding known lesions and a surrounding margin. These are not healthy control subjects.
- 64×64 coronal patches at 1 mm; frozen evaluation targets and masks; single samples with seeds 909 and 910. No sample averaging or output sharpening.
- Training expanded to 192 subjects. RF training coverage expanded to 2,304 coronal planes, 112,775 eligible crops, MNI y −80 through +52 mm. Three of the original four examples lay outside the earlier five-plane training range.
- The four earlier examples are retained, including the very large partial mask. Sixteen additional cortical patches come from eight subjects outside training and checkpoint selection. They were selected from the existing validation pool; the reserved 196-subject test manifest was not touched.
- The initial final-evaluation masks sometimes met crop boundaries. Those results are retained separately. The centered check uses the same held-out subjects with radius-10 and radius-16 holes centered in a 64-pixel patch, leaving a full visible rim. Each hole has at least 90% intensity foreground and hides both low- and high-intensity tissue. All methods use identical inputs.
- Target intensities inside each mask are absent from actual RF conditioning. A target-perturbation check left inference conditions bit-identical. Known pixels outside masks remain identical, and all final predictions are finite.

## Centered cortical results

Two-seed means over 16 patches from eight held-out subjects. The legacy baseline has a smaller architecture and less training data, so the old/new comparison measures the combined setup change, not an isolated architecture effect. The RF control and critic variant have matched initialization, training inputs, noise, updates and inference steps; the critic variant uses extra computation for its auxiliary loss.

| Actual inpainting method | Masked MAE ↓ | Masked SSIM ↑ | Gradient error ↓ |
|---|---:|---:|---:|
| Original pixel RF | 0.3349 | 0.2246 | 0.1202 |
| Revised early-time latent RF | 0.2706 | 0.3008 | 0.1027 |
| Matched RF control, two steps | 0.2701 | 0.3014 | 0.1065 |
| RF with auxiliary critic | 0.2858 | 0.2926 | 0.1114 |
| Deterministic blurry comparison | 0.2368 | 0.3802 | 0.0974 |

The deterministic comparison remains blurry despite better SSIM. It is not an eligible final generator under the rectified-flow requirement. The AE oracle is also ineligible. Higher SSIM alone is insufficient to pass the requested cortical-detail check.

## What was tried

1. A MONAI AutoencoderKL representation (8×16×16 codes) plus a MONAI DiffusionModelUNet RF. Both were originally trained from scratch on this project’s data. The autoencoder preserves held-out folds; the first latent RF still produced incorrect anatomy.
2. Reducing source noise from 0.65 to 0.1 did not solve the problem.
3. A balanced seven-component latent basis, robust decoder training and broader coronal coverage did not improve generated validation sufficiently. This variant is rejected, not promoted as a fix.
4. The source-dependent latent RF exhibited sampling drift: on the original latent checkpoint, generated validation SSIM was 0.291 with one Euler step versus 0.212 with 32 Euler steps, using the robust decoder. Teacher-forced intermediate paths improved as true target information was introduced; those diagnostics are explicitly not inpainting examples.
5. More early-time training (one third of times exactly zero, the remainder squared-uniform), a frozen robust decoder and validation of actual generated images improved the measured RF result. One-step RF still produces overly smooth or misplaced structures.
6. A training-only spectral-normalized MRI critic and feature loss were compared with a matched RF control using differentiable two-step generation. The primary objective remains straight-path RF velocity regression. Its results do not justify adding this component to production.

These were separate bounded local tests, not one 15-minute total review. They provide negative and partial results, not evidence that a new long training run will succeed.

## Interpretation and next quality gate

The decoder can represent the detail, while the learned flow does not generate the correct missing anatomical representation. More reconstruction pressure improves average intensity error but continues to erase folds. Extra stochastic texture or a critic can introduce structures without improving their anatomical placement. Training coverage and rollout drift were real issues in this 2D setup; correcting them did not resolve cortical recovery.

A further RF experiment should establish a stronger healthy-anatomy generative prior and assess generated trajectories, rather than rely on image-regression losses or an AE oracle. This remains an untested next experiment, not a promised fix. An RF-based precedent learns an unconditional healthy-anatomy prior and imposes observed data during inference: [Hadzic et al., 2026](https://arxiv.org/abs/2610.08983). Its reported results do not validate this project’s implementation.

Before any long 96³ run, require improved fold continuity and normal tissue appearance on fixed masked cortical examples, adequate full ROI context, preserved known tissue, and independent subject-level checks. No 3D 96³ model was trained or launched in this review.

Local review artifacts: `/home/ajoshi/Documents/Codex/2026-11-24/i-x20/outputs/2d_rf_quality_review_centered/`. MRI-bearing artifacts are not stored in this repository.

## Review artifacts

- `actual_rf_vs_original.png`: all four original cases, original pixel RF and revised actual RF.
- `ae_oracle_explained.png`: original, full-input AE reconstruction, actual masked-input RF.
- `fresh_cortical_01_04.png` through `fresh_cortical_13_16.png`: all centered held-out examples.
- `fresh_critic_01_04.png` through `fresh_critic_13_16.png`: matched critic/control comparisons.
- `results.json`, `checkpoint_hashes.json`, `fresh_cortical_sources.json`: exact metrics, model identity and sampling metadata.
- `code/`: retained local-review scripts. They retain paths to this workstation’s authorized data and earlier experiment caches; they are experiment records, not a new remote-training CLI. Earlier runs used wall-clock budgets and scripts evolved during debugging, so reruns need not reproduce identical training streams. Hashed checkpoints and saved inputs identify the evaluation precisely.

The earlier partial-context evaluation remains in the sibling `2d_rf_quality_review` directory. Checkpoints and MRI-bearing input/prediction tensors remain local and are not added to GitHub.
