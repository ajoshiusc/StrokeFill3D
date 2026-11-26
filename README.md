# StrokeFill3D

MONAI research code for 3D T1 MRI stroke-lesion inpainting using a **normal-code rectified-flow prior**. The selected workflow trains a new 3D autoencoder and a complete normal-code RF prior, then fits the initial RF noise to the observed MRI during inpainting.

The RTX 4090 profile uses **96×96×96 MRI patches** and **48×48×48 latent codes**, with position conditioning and generated validation. It ports the [improved small 2D experiment](docs/NORMAL_CODE_RF_2D_20261126.md). Local execution is tested; trained 3D anatomical quality remains to be evaluated. Generated anatomy is an estimate, not recovered patient ground truth.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
# Install a CUDA-enabled PyTorch 2.6 wheel appropriate to your driver first.
python -m pip install -e .
```

The working local environment uses PyTorch 2.6.0, MONAI 1.5.2 and CUDA 12.4. Reuse the existing remote environment and run `python -m pip install -e .` after pulling to install the declared Pillow dependency.

## Train on the RTX 4090

Full commands, stage details, resume instructions and inference are in [the normal-prior 4090 guide](docs/NORMAL_PRIOR_4090.md).

```bash
git pull
source .venv/bin/activate
python -m pip install -e .
bash scripts/train_normal_prior_4090.sh /tmp/normal_prior_4090_preflight --preflight

# Run inside tmux. Each host/attempt gets its own models and code cache.
RUN="/deneb_disk/data/stroke_inpainting/runs/normal_prior_4090_$(hostname -s)_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN/logs"
set -o pipefail
bash scripts/train_normal_prior_4090.sh "$RUN" 2>&1 | tee "$RUN/logs/train.log"
```

The pipeline trains the **3D AE from scratch**, checks its fine-detail reconstruction, freezes/calibrates it, builds a training-only code bank, and trains the **3D RF from scratch**. It stops if AE quality fails its gate. Old 2D or mask-conditioned RF weights cannot initialize this architecture. Continue an interrupted run with the same `RUN` and `--resume`.

The RF learns straight paths from independent Gaussian noise to complete nonlesional codes, conditioned only on MNI position. Source fitting backpropagates through the frozen RF and decoder using visible intensities and visible-pair gradients. This retains RF as the generator; there is no regression completion head, critic, retrieved target patch, output sharpening or Gaussian tile averaging.

## Data and validation

Authorized ATLAS3 scans are already available under `/deneb_disk/ATLAS3/`; manifests live under `/deneb_disk/data/stroke_inpainting/manifests/`. For a new installation, follow [the data guide](docs/DATA.md) and build manifests with `python scripts/build_atlas3_training_manifests.py`. Dataset access terms still apply; this repository contains no MRI data or model weights.

Targets are nonlesional regions from stroke scans. Full training crops must avoid the real lesion and a three-voxel exclusion margin. The existing split uses 1,172 training scans from 44 sites, 84 validation scans from five sites, and a reserved 196-scan test set from six sites. Reserved test MRIs are not loaded by this workflow. Training-site lesion masks supply shapes for fixed cortical validation holes; the RF itself learns complete normal codes.

AE oracle reconstruction and actual inpainting are reported separately. RF checkpoints are selected using actual generated validation, with a more expensive two-seed evaluation at the end. Every preview has corresponding NIfTI outputs and metrics. Outside-mask values are preserved exactly, and hidden intensities are excluded from inference normalization and fitting.

## Tests

```bash
python -m unittest discover -s tests -v
python -m unittest discover -s experiments/normal_code_rf_2d -p 'test_*.py' -v
bash scripts/train_normal_prior_4090.sh /tmp/normal_prior_preflight --preflight
```

The real 96³ preflight includes configured RF batch 2, AE/RF backward passes, checkpoint reload, source gradients and NIfTI IO. It passed locally on the RTX 2060. It uses untrained weights and demonstrates execution, not realistic 3D output quality.

## Earlier experiments

The [portable 2D normal-code RF prototype](experiments/normal_code_rf_2d/README.md), [2D follow-up report](docs/NORMAL_CODE_RF_2D_20261126.md), and [earlier quality review](docs/2D_QUALITY_GATE_20261126.md) preserve the evidence motivating this port, including failures. The [older anatomical 4090 workflow](docs/ANATOMICAL_4090.md), legacy `strokefill-train` / `strokefill-infer` entry points, pixel-RF configurations and deterministic SwinUNETR baseline remain available for reproducibility. They do **not** train the selected normal-prior model. Use `scripts/train_normal_prior_4090.sh` for the new run.

Licensed under Apache-2.0. Research software; not validated for clinical decisions.
