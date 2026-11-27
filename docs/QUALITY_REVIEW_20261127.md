# Cortical inpainting quality review

**A new long 96³ training run is not yet qualified.** Software checks and an accurate autoencoder reconstruction do not establish realistic inpainting.

## Correctness improvements

External scans now have a documented skull-stripping, lesion-excluded N4, rigid/affine alignment and mask-safe intensity normalization workflow. The inverse intensity transform is linear; the earlier clipped histogram inverse exaggerated bright intensities. These changes address preprocessing errors but do not fix the generator's missing cortical detail. See [external-data inference](EXTERNAL_DATA.md).

Validation measures the generated interior separately from the copied observations. Interior SSIM uses a three-voxel erosion so its comparison windows stay within the hole. Gradient direction and amplitude help distinguish plausible structure from blur or excess texture. Undefined interior scores are reported explicitly. Checkpoint metadata identifies the exact loaded weights, and tile caches are invalidated after relevant code changes.

## Controlled local experiments

These are 2D 64×64 development experiments, not evidence of successful 3D training. Normal-appearing training patches come from 192 participants; annotated injuries and a three-voxel 3D margin are excluded. They are not a healthy-control cohort. Actual inpainting never receives intensities hidden by its mask. Autoencoder oracles receive the complete target and are reported separately.

The following rectified-flow variants have not demonstrated the required improvement in sulcal continuity and normal tissue appearance:

- Adding visible-image conditioning to the existing latent RF.
- Image-space RF with contralateral conditioning or a contralateral source.
- A coarse conditional estimate followed by RF, including residual-state transport.
- Frozen autoencoder feature supervision and a training-only image critic.
- A smaller learned latent representation. Its intact-target decoder retains fine detail, but quick generated validation remains worse than the existing normal-code RF after 20,000 updates.

Increasing integration steps and changing observation-projection strength have also failed to resolve the issue. Solver diagnostics with a source optimized for one integration scheme are sensitivity checks, not matched reoptimized inference comparisons. Different representations use reproducible seed identifiers; they cannot share identical noise arrays when state dimensions differ.

These negative experiments do not justify replacing the production model or adding their components to the standard training configuration. They also do not prove that rectified flow cannot solve the task: the experiment budgets, architectures and data coverage are limited. A one-image overfit can reproduce sharp folds, whereas multi-subject generated validation remains poor. Low velocity-regression loss and accurate decoding are insufficient qualification criteria.

The final compact-code comparison used all four pre-existing cortical development crops and seeds 909 and 910, with 16 Euler steps and 120 source-fitting updates for both models:

| Actual inpainting | Masked SSIM ↑ | Interior SSIM ↑ | Interior gradient cosine ↑ |
|---|---:|---:|---:|
| Existing normal-code RF | 0.378 | 0.246 | 0.201 |
| Compact-code RF, validation-selected update 4,000 | 0.366 | 0.227 | 0.227 |
| Compact-code RF, fixed update 20,000 | 0.359 | 0.189 | 0.199 |

The compact AE oracle scored 0.957 masked SSIM with access to the complete target; it is not an eligible inpainting comparison. The combined codec/training setup differs from the baseline, so this is not an isolated compression ablation. Different state dimensions also prevent identical noise arrays; seed identifiers are reproducible. Neither candidate met the predefined improvement thresholds or the visual cortical-detail requirement. No new long 3D run was launched.

## Readiness criterion

A candidate must improve actual masked-input generation on fixed cortical examples with multiple seeds, preserve observed voxels exactly, and show continuous sulci and gyral boundaries without bright islands or spongy texture. Subject-level confirmation must follow model selection; evaluation examples used to choose another variant become development data. Nonlesional patches from a single external participant cannot establish cross-dataset generalization. The reserved test manifest remains unused.

Only a successful small experiment should lead to a new 3D profile, a local 96³ execution/resume/inference check and another long training run. Neither those execution checks nor favorable 2D scores guarantee anatomically accurate 3D fills. For real lesions, the patient's original hidden folding pattern is unavailable.

Experiment images, inputs and checkpoints remain outside the public repository. The standard model and existing run compatibility are preserved; no failed experimental generator is promoted as a quality fix.
