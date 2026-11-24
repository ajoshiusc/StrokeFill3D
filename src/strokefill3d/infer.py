"""Generate a stochastic inpainting sample with conditional rectified flow."""
from __future__ import annotations
import argparse
import nibabel as nib
import numpy as np
import torch
from monai.inferers import sliding_window_inference
from monai.transforms import Compose, LoadImage, Orientation, ScaleIntensityRangePercentiles, Spacing
from .model import RectifiedFlowUNet


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True); p.add_argument("--mask", required=True)
    p.add_argument("--checkpoint", required=True); p.add_argument("--output", required=True)
    p.add_argument("--steps", type=int, default=24)
    p.add_argument("--roi-size", type=int, nargs=3, default=(64,64,64))
    p.add_argument("--overlap", type=float, default=.5)
    p.add_argument("--seed", type=int, default=None, help="Set for repeatable samples; omit for varied completions")
    p.add_argument("--time-embedding-scale", type=float, default=1000.0)
    args = p.parse_args()
    if args.steps < 1:
        raise ValueError("--steps must be at least 1")
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
    context = image_transform(args.image).float().unsqueeze(0)
    mask = (mask_transform(args.mask) > 0).float().unsqueeze(0)
    if context.shape[2:] != mask.shape[2:] or not np.allclose(context.affine.cpu().numpy(), mask.affine.cpu().numpy(), atol=1e-4):
        raise ValueError("Image and lesion mask do not align after RAS orientation and 1 mm resampling")
    if not torch.any(mask > 0):
        raise ValueError("Lesion mask is empty")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.seed is not None:
        generator = torch.Generator(device=device).manual_seed(args.seed)
        noise = torch.randn(context.shape, generator=generator, device=device)
    else:
        noise = torch.randn(context.shape, device=device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model_cfg = checkpoint.get("model_config", {})
    model = RectifiedFlowUNet(**model_cfg).to(device)
    model.load_state_dict(checkpoint["model"]); model.eval()
    context = context.to(device); mask = mask.to(device)
    evolving = torch.where(mask > 0, noise, context)
    with torch.no_grad():
        for index in range(args.steps):
            t = index / args.steps
            model_input = torch.cat((evolving, context, mask), dim=1)
            t_scaled = t * args.time_embedding_scale
            def predict(patches):
                times = torch.full((patches.shape[0],), t_scaled, device=patches.device)
                return model.backbone(x=patches, timesteps=times)
            velocity = sliding_window_inference(model_input, tuple(args.roi_size), 1,
                                                predict, overlap=args.overlap)
            evolving = torch.where(mask > 0, evolving + velocity * (1.0 / args.steps), context)
    normalized_out = torch.where(mask > 0, evolving.clamp(-1, 1), context).squeeze().cpu().numpy().astype(np.float32)
    out = ((normalized_out + 1.0) * .5 * (hi - lo) + lo).astype(np.float32)
    affine = context.affine.cpu().numpy()
    header = nib.Nifti1Header(); header.set_data_dtype(np.float32)
    nib.save(nib.Nifti1Image(out, affine, header), args.output)

if __name__ == "__main__": main()
