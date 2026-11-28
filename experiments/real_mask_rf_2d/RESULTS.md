# Bounded 2D result

The fine-tune ran for 3,000 updates on the local RTX 2060 in about 5 minutes.
It started from the 20,000-update native-image RF checkpoint. Targets came from
the complete lesion-excluded ATLAS3 patch bank (3-voxel 3D exclusion margin).
Medium-case binary contour slices were used for training; hard-case contours
were held out. No lesion-case T1 intensities were read.

The fixed paired evaluation used four complete held-out healthy T1 patches,
four held-out irregular contours, and two identical initial-noise seeds for all
comparisons. Metrics below are means over those seeds. The target is known in
this synthetic benchmark; these scores do not establish the correctness of
inpainting on an actual lesion.

| Model and sampler | Masked MAE ↓ | Masked gradient MAE ↓ | Masked SSIM ↑ |
|---|---:|---:|---:|
| Ellipse-trained, legacy Euler 20 | 0.2119 | **0.1142** | 0.4950 |
| Real-mask fine-tune, legacy Euler 20 | **0.2045** | 0.1144 | **0.5261** |
| Ellipse-trained, endpoint Euler 20 | 0.2126 | 0.1144 | 0.4932 |
| Real-mask fine-tune, endpoint Euler 20 | 0.2051 | 0.1146 | 0.5245 |
| Ellipse-trained, endpoint Heun 20 | 0.2165 | 0.1168 | 0.4859 |
| Real-mask fine-tune, endpoint Heun 20 | 0.2093 | 0.1169 | 0.5167 |
| Real-mask fine-tune, endpoint Heun 40 | 0.2095 | 0.1174 | 0.5161 |

Real-contour fine-tuning improved MAE by about 3.5% and SSIM by 0.031 over the
legacy Euler baseline. Gradient error was essentially unchanged. The tested
Heun endpoint sampler and extra Heun steps did not improve the results, so the
legacy 20-step Euler sampler remains the selected generation path for now.

The result supports continuing with complete clean targets and real mask
geometry. It is a modest 2D signal on four paired patches, not enough evidence
to replace or restart the long 3D model. The 3,000-update experimental weights,
full comparison image, sampler metrics, and run protocol are saved beside the
workspace outputs; the 3D production checkpoint was not changed.
