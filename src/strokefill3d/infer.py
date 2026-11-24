"""Apply a trained model to a T1 volume and lesion mask."""
from __future__ import annotations
import argparse
import nibabel as nib
import numpy as np
import torch
from monai.inferers import sliding_window_inference
from monai.transforms import Compose, LoadImage, Orientation, ScaleIntensityRangePercentiles, Spacing
from .model import StrokeFillNet


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True); p.add_argument("--mask", required=True)
    p.add_argument("--checkpoint", required=True); p.add_argument("--output", required=True)
    p.add_argument("--patch-size", type=int, nargs=3, default=(96,96,96))
    p.add_argument("--feature-size", type=int, default=48)
    p.add_argument("--overlap", type=float, default=.5)
    args = p.parse_args()
    raw = np.asarray(nib.load(args.image).dataobj, dtype=np.float32)
    finite = raw[np.isfinite(raw)]
    if finite.size == 0:
        raise ValueError("Input image has no finite voxels")
    lo, hi = np.percentile(finite, [1, 99])
    image_transform = Compose([LoadImage(image_only=True, ensure_channel_first=True),
                               Orientation(axcodes="RAS"), Spacing(pixdim=(1.,1.,1.), mode="bilinear"),
                               ScaleIntensityRangePercentiles(lower=1, upper=99, b_min=-1, b_max=1, clip=True)])
    mask_transform = Compose([LoadImage(image_only=True, ensure_channel_first=True),
                              Orientation(axcodes="RAS"), Spacing(pixdim=(1.,1.,1.), mode="nearest")])
    x = image_transform(args.image).float().unsqueeze(0)
    m = (mask_transform(args.mask) > 0).float().unsqueeze(0)
    if x.shape[2:] != m.shape[2:]:
        raise ValueError("Image and lesion mask do not align after RAS orientation and 1 mm resampling")
    x_masked = x.clone(); x_masked[m > 0] = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = StrokeFillNet(tuple(args.patch_size), args.feature_size, use_checkpoint=False).to(device)
    model.load_state_dict(state.get("model", state)); model.eval()
    with torch.no_grad():
        # Combine both conditions into channels so SlidingWindowInferer windows them identically.
        inputs = torch.cat((x_masked, m), dim=1).to(device)
        pred = sliding_window_inference(inputs, tuple(args.patch_size), 1,
            lambda patch: model(patch[:, :1], patch[:, 1:2]), overlap=args.overlap)
    out_norm = torch.where(m > 0, pred.cpu(), x).squeeze().numpy().astype(np.float32)
    out = ((out_norm + 1.0) * 0.5 * (hi - lo) + lo).astype(np.float32)
    meta = x.meta
    affine = np.asarray(meta.get("affine", np.eye(4)))
    header = nib.Nifti1Header(); header.set_data_dtype(np.float32)
    nib.save(nib.Nifti1Image(out, affine, header), args.output)

if __name__ == "__main__": main()
