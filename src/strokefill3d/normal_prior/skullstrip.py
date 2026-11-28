"""SynthStrip v1 inference using only PyTorch, NiBabel and SciPy.

Implements the published network and preprocessing convention for the official
SynthStrip v1 weights. No FreeSurfer executable or surfa installation is used.
Reference: Hoopes et al., NeuroImage 260 (2022), 119474.
https://surfer.nmr.mgh.harvard.edu/docs/synthstrip/
"""

import nibabel as nib
import numpy as np
import torch
from nibabel.processing import resample_from_to
from scipy.ndimage import binary_fill_holes, label
from torch import nn

from .assets import get_asset, sha256


class ConvBlock(nn.Module):
    def __init__(self, incoming, outgoing, activate=True):
        super().__init__()
        self.conv = nn.Conv3d(incoming, outgoing, 3, padding=1)
        self.activation = nn.LeakyReLU(0.2) if activate else nn.Identity()

    def forward(self, value):
        return self.activation(self.conv(value))


class StripModel(nn.Module):
    """Fixed SynthStrip v1 topology; parameter names match published weights."""

    def __init__(self):
        super().__init__()
        features = [16, 32, 64, 64, 64, 64, 64]
        self.encoder = nn.ModuleList()
        incoming = 1
        skips = []
        for outgoing in features[:-1]:
            self.encoder.append(
                nn.Sequential(
                    ConvBlock(incoming, outgoing), ConvBlock(outgoing, outgoing)
                )
            )
            skips.append(outgoing)
            incoming = outgoing
        self.decoder = nn.ModuleList()
        for outgoing, skip in zip(features[:0:-1], skips[::-1]):
            self.decoder.append(
                nn.Sequential(
                    ConvBlock(incoming, outgoing), ConvBlock(outgoing, outgoing)
                )
            )
            incoming = outgoing + skip
        self.remaining = nn.Sequential(
            ConvBlock(incoming, 16), ConvBlock(16, 16), ConvBlock(16, 1, activate=False)
        )

    def forward(self, value):
        skips = []
        for block in self.encoder:
            value = block(value)
            skips.append(value)
            value = nn.functional.max_pool3d(value, 2)
        for block, skip in zip(self.decoder, skips[::-1]):
            value = block(value)
            value = nn.functional.interpolate(value, scale_factor=2, mode="nearest")
            value = torch.cat((value, skip), dim=1)
        return self.remaining(value)


def conform(image):
    """LIA, 1-mm voxels, positive-data bounding box, then centered /64 padding.

    Preserve obliquity when changing voxel size. SynthStrip uses the FreeSurfer
    shape/2 physical-center convention, rather than (shape-1)/2.
    """
    orientation = nib.orientations.ornt_transform(
        nib.orientations.io_orientation(image.affine),
        nib.orientations.axcodes2ornt(("L", "I", "A")),
    )
    oriented = image.as_reoriented(orientation)
    affine = oriented.affine.copy()
    spacing = nib.affines.voxel_sizes(affine)
    data = np.asarray(oriented.dataobj, dtype=np.float32)
    if not np.allclose(spacing, 1.0, atol=1e-5, rtol=0):
        shape = np.ceil(np.asarray(data.shape) * spacing).astype(int)
        center = nib.affines.apply_affine(affine, np.asarray(data.shape) / 2)
        affine[:3, :3] /= spacing
        affine[:3, 3] = center - affine[:3, :3] @ (shape / 2)
        resized = resample_from_to(oriented, (tuple(shape), affine), order=0)
        data = np.asarray(resized.dataobj, dtype=np.float32)
    indices = np.where(data > 0)
    if not len(indices[0]):
        raise ValueError("Brain extraction requires positive MRI intensities")
    lo = np.array([a.min() for a in indices])
    hi = np.array([a.max() + 1 for a in indices])
    data = data[tuple(slice(a, b) for a, b in zip(lo, hi))]
    affine[:3, 3] = nib.affines.apply_affine(affine, lo)
    target = np.clip(np.ceil(np.array(data.shape) / 64) * 64, 192, 320).astype(int)
    delta = (target - np.array(data.shape)) / 2
    low, high = np.floor(delta).astype(int), np.ceil(delta).astype(int)
    padded = np.pad(data, list(zip(np.maximum(low, 0), np.maximum(high, 0))))
    crop_low = np.maximum(-high, 0)
    crop_high = np.array(padded.shape) - np.maximum(-low, 0)
    data = padded[tuple(slice(a, b) for a, b in zip(crop_low, crop_high))]
    affine[:3, 3] = nib.affines.apply_affine(affine, crop_low - np.maximum(low, 0))
    return nib.Nifti1Image(data, affine)


def extract_brain(image, weights=None, cache_dir=None, device="cpu", threads=4):
    path = weights or get_asset("synthstrip", cache_dir)
    model = StripModel()
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval().requires_grad_(False)
    conformed = conform(image)
    data = np.asarray(conformed.dataobj, dtype=np.float32).copy()
    data -= data.min()
    scale = float(np.percentile(data, 99))
    if scale <= 0:
        raise ValueError("Brain extraction input has no usable intensity variation")
    data = np.clip(data / scale, 0, 1)
    old_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(threads)
        with torch.inference_mode():
            distance = model(torch.from_numpy(data[None, None]).to(device))
            distance = distance[0, 0].cpu().numpy()
    finally:
        torch.set_num_threads(old_threads)
    native_distance = resample_from_to(
        nib.Nifti1Image(distance, conformed.affine), image, order=1, cval=100
    )
    components, count = label(np.asarray(native_distance.dataobj) < 1.0)
    if not count:
        raise RuntimeError("SynthStrip produced an empty brain mask")
    sizes = np.bincount(components.ravel())
    sizes[0] = 0
    mask = binary_fill_holes(components == sizes.argmax())
    return mask, {"method": "SynthStrip v1 in PyTorch", "weights_sha256": sha256(path)}
