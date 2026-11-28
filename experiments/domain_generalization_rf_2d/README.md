# Mixed-domain 2D rectified-flow experiment

This experiment checks whether adding clean T1 patches from other acquisition
domains improves rectified-flow inpainting. It retains the existing RF model,
real irregular hide masks, and lesion-free target rule.

Each target patch is complete and excludes the lesion mask plus a 5-mm 3D
margin. Binary lesion or resection masks provide contour shapes only. They do
not supply target intensities. Images are normalized within their brain masks
using the 1st and 99th percentiles.

The split holds out the two difficult EPISURG/EPIBIOS cases and holds out ARC
participants as groups, so repeat scans from one person cannot be split across
training and evaluation. The paired evaluation hides real irregular contours
inside complete normal-appearing patches. It is a synthetic benchmark with a
known target; it is not a measure of correctness on actual lesion voxels.

Two fine-tunes start from the same real-mask 2D RF checkpoint and use the same
training draws, noise, masks, optimizer, and update count:

- `standard`: mixed-domain targets with the existing mild gamma/gain changes.
- `hist_bias`: the same training plus a monotone histogram warp and a smooth
  multiplicative bias field.

The evaluation compares both with the ATLAS-only checkpoint on held-out
EPISURG, EPIBIOS, ARC, and ATLAS patches. All methods use the existing Euler20
sampler. Per-domain metrics and an image panel are produced by `evaluate.py`.

The data-preparation command takes paths to MNI-aligned T1, brain, and lesion
files, plus an explicit participant-level ARC split. No specific participant
IDs or local filesystem paths are built into the code.
