# Real cortical stroke example

This page shows how to run the 3D normal-prior model on a T1 scan with a real cortical stroke mask. The figure is one example of the model's output; it does not show ground-truth anatomy inside the lesion.

![Masked T1, original scan, and model inpainting in three orthogonal views](images/real_stroke_example.png)

The three columns show the masked input with the lesion outline, the original scan, and the inpainted result. The figure uses one intensity window for all images and has no sharpening. The lesion mask was dilated by two voxels before filling. Values outside that expanded mask are copied directly from the input.

The example used a saved 96³ model checkpoint, seed 909, 32 flow steps, and 240 source-fitting updates. Those last two values are higher than the command defaults of 16 flow steps and 120 updates. Increasing them reduced the fitting error on visible voxels, but did not make the cortical folds clearly sharper. The tissue hidden by a real stroke has no known healthy reference, so the image cannot confirm that the generated anatomy is correct.

## Run inference

The input T1 and stroke mask must already be registered to the same 1-mm MNI grid. This command does not register or resample them. It checks that they share a grid and stops if a lesion component cannot fit inside the trained 96³ patch with enough surrounding context.

Activate the project environment, then set the run directory and file paths:

```bash
cd /path/to/StrokeFill3D
source .venv/bin/activate

RUN=/path/to/training_run
T1=/path/to/MNI_T1.nii.gz
MASK=/path/to/MNI_lesion_mask.nii.gz
```

Run the inpainting command:

```bash
python -m strokefill3d.normal_prior.infer \
  --run "$RUN" \
  --checkpoint "$RUN/prior/best.pt" \
  --image "$T1" \
  --mask "$MASK" \
  --output /path/to/inpainted_T1.nii.gz \
  --mni
```

`--mni` is required. It confirms that the inputs are already in the expected MNI grid; it does not transform them. By default, inference uses CUDA, 16 flow steps, 120 source-fitting updates, seed 909, and two-voxel unconditional mask dilation. To try the higher inference budget shown in the example, add:

```bash
  --flow-steps 32 --iterations 240
```

The run directory must contain `codec.pt` and the checkpoint trained with that codec. The output NIfTI stays in the source T1 intensity units. A JSON sidecar records the checkpoint and inference settings. The repository includes only the rendered example figure; scans, lesion masks, and trained weights are not included.
