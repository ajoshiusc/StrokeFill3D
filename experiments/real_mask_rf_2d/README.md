# Real-mask 2D rectified-flow experiment

This bounded experiment keeps the conditional rectified-flow model and changes
the mask training distribution. Each target is a complete ATLAS3 T1 patch from
the lesion-excluded training bank (including its 3-voxel 3D lesion margin).
Training and held-out lesion masks contribute binary 2D contour geometry only;
the lesion cases' T1 intensities are never read. Medium-case contours form the
training pool; hard-case contours are held out for the synthetic paired test.

The fine-tune starts from the existing 2D native-image RF checkpoint trained on
complete targets with ellipse masks. Evaluation uses the same clean targets,
held-out irregular masks, and noise seed for the base model and fine-tuned model.
It reports masked intensity, gradient, and SSIM metrics. The sampler comparison
also checks Heun integration over the trained time interval with a clean endpoint
estimate against the prior Euler sampler's extrapolation past that interval.

Example command (paths are supplied by the operator):

```bash
python experiments/real_mask_rf_2d/train.py \
  --pairs /path/to/lesion_excluded_pairs.pt \
  --initial /path/to/clean_mri_best.pt \
  --development /path/to/heldout_clean_patches.pt \
  --train-mask /path/to/medium_case_mask_a.nii.gz \
  --train-mask /path/to/medium_case_mask_b.nii.gz \
  --heldout-mask /path/to/hard_case_mask_a.nii.gz \
  --heldout-mask /path/to/hard_case_mask_b.nii.gz \
  --output /path/to/experiment_output \
  --steps 3000 --batch 8
```

The experiment is deliberately 2D and does not alter the production 3D model
or any 4090-host run. The generated checkpoint is an experimental fine-tune.
