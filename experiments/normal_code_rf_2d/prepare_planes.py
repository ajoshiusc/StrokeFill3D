"""Build a lesion-excluded, MNI-aligned coronal plane bank from a training CSV."""

from pathlib import Path
import argparse, csv, json, random
import numpy as np, nibabel as nib, torch
from scipy.ndimage import binary_dilation


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--subjects", type=int, default=192)
    p.add_argument("--seed", type=int, default=90231)
    args = p.parse_args()
    random.seed(args.seed)
    with args.manifest.open(newline="") as f:
        rows = list(csv.DictReader(f))
    random.shuffle(rows)
    rows = rows[: args.subjects]
    planes = []
    exclusions = []
    world_y = []
    subject_index = []
    candidates = []
    sources = []
    xs = np.arange(0, 133, 12)[:, None]
    zs = np.arange(38, 123, 12)[None, :]

    def boxes(a):
        s = np.pad(a.astype(np.float64).cumsum(0).cumsum(1), ((1, 0), (1, 0)))
        return s[xs + 64, zs + 64] - s[xs, zs + 64] - s[xs + 64, zs] + s[xs, zs]

    expected = np.diag([1.0, 1.0, 1.0, 1.0])
    expected[:3, 3] = [-98, -134, -72]
    for sid, row in enumerate(rows):
        nii = nib.load(row["healthy_t1"])
        les = nib.load(row["exclude_mask"])
        if nii.shape != (197, 233, 189) or not np.allclose(
            nii.affine, expected, atol=0.001
        ):
            raise ValueError(
                "This 2D experiment requires the canonical ATLAS3 1-mm MNI grid"
            )
        if les.shape != nii.shape or not np.allclose(
            les.affine, nii.affine, atol=0.001
        ):
            raise ValueError("T1/lesion grids differ")
        a = np.asarray(nii.dataobj, dtype=np.float32)
        lo, hi = np.percentile(a, [1, 99])
        a = np.clip((a - lo) / max(hi - lo, 1e-6) * 2 - 1, -1, 1).astype(np.float32)
        excluded = binary_dilation(np.asarray(les.dataobj) > 0, iterations=3)
        for y in range(54, 191, 12):
            plane = a[:, y, :].copy()
            ex = excluded[:, y, :].copy()
            idx = len(planes)
            fg = boxes(plane > -0.95) / 4096
            bad = boxes(ex)
            for xi, zi in np.argwhere((fg >= 0.08) & (bad == 0)):
                candidates.append((idx, int(xs[xi, 0]), int(zs[0, zi])))
            planes.append(torch.from_numpy(plane))
            exclusions.append(torch.from_numpy(ex))
            world_y.append(y - 134)
            subject_index.append(sid)
        sources.append(
            dict(source=row["healthy_t1"], site=row.get("site", ""), split="train")
        )
        if (sid + 1) % 16 == 0:
            print(f"Prepared {sid + 1}/{len(rows)} subjects", flush=True)
    data = dict(
        planes=torch.stack(planes)[:, None],
        exclusions=torch.stack(exclusions)[:, None],
        world_y=torch.tensor(world_y, dtype=torch.float32),
        subject_index=torch.tensor(subject_index),
        candidates={"train": candidates},
        sources=sources,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, args.output)
    args.output.with_suffix(".sources.json").write_text(json.dumps(sources, indent=2))
    print(
        f"Saved {len(planes)} planes and {len(candidates)} valid crops to {args.output}"
    )


if __name__ == "__main__":
    main()
