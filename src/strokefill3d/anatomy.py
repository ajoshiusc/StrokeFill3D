"""Observable MNI-space anatomical cues shared by training and inference."""

from __future__ import annotations

import numpy as np
import torch
from scipy.ndimage import gaussian_filter, map_coordinates

ANATOMY_CHANNELS = 7
ANATOMY_VERSION = "mni_observed_v1"


def check_mni(affine, shape):
    """Require the axis-aligned, 1 mm MNI grid used by this profile.

    Geometry alone cannot establish registration: callers must supply an MNI image.
    """
    affine = np.asarray(affine, dtype=np.float64)
    if not np.allclose(affine[:3, :3], np.eye(3), atol=1e-3):
        raise ValueError(
            "Anatomical conditioning requires an axis-aligned RAS 1 mm MNI image"
        )
    ends = affine[:3, 3] + np.asarray(shape) - 1
    if not (affine[0, 3] < -40 and ends[0] > 40):
        raise ValueError("MNI volume must span both hemispheres about world x=0")


def anatomy_features(volume, missing, affine, crop=None, excluded=None):
    """Return mirror intensity/validity, xyz, coarse intensity/validity (7 channels).

    `volume` is full-volume normalized T1. Hidden synthetic/real lesion values are
    removed BEFORE reflection or smoothing. Opposite-side anatomy is a cue, not a
    recovered healthy ground truth. Missing and out-of-brain mirror cues are zero.
    """
    a = np.asarray(volume, dtype=np.float32)
    check_mni(affine, a.shape)
    missing = np.asarray(missing, dtype=bool)
    if missing.shape != a.shape:
        raise ValueError("Conditioning image and mask shapes differ")
    observed = (~missing) & np.isfinite(a) & (a > -0.9999)
    if excluded is not None:
        observed &= ~np.asarray(excluded, dtype=bool)
    visible = np.where(observed, a, 0).astype(np.float32)
    crop = crop or tuple(slice(0, n) for n in a.shape)
    indices = np.stack(
        np.meshgrid(
            *[np.arange(s.start or 0, s.stop, dtype=np.float32) for s in crop],
            indexing="ij",
        )
    )
    affine = np.asarray(affine, dtype=np.float32)
    world = indices + affine[:3, 3, None, None, None]
    reflected = indices.copy()
    reflected[0] = -world[0] - affine[0, 3]
    # Linear validity requires all contributing mirror voxels to be observed.
    validity = map_coordinates(
        observed.astype(np.float32),
        reflected,
        order=1,
        mode="constant",
        cval=0,
        prefilter=False,
    )
    mirror = map_coordinates(
        visible, reflected, order=1, mode="constant", cval=0, prefilter=False
    )
    valid = validity > 0.999
    mirror = np.where(valid, mirror, 0)
    support = gaussian_filter(observed.astype(np.float32), sigma=6)
    coarse = gaussian_filter(visible, sigma=6) / np.maximum(support, 1e-5)
    coarse = np.where(support > 0.01, coarse, 0)
    features = np.concatenate(
        [
            mirror[None],
            valid[None],
            np.clip(world / 100.0, -2, 2),
            coarse[crop][None],
            support[crop][None],
        ],
        axis=0,
    )
    return torch.from_numpy(features.astype(np.float32))
