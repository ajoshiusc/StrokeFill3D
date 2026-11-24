"""Pair ATLAS3 preprocessed T1/masks and reserve whole sites for holdout."""
from __future__ import annotations
import argparse
import csv
import random
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="/deneb_disk/ATLAS3/ATLAS3_Training_Preprocessed")
    p.add_argument("--output-dir", default="/deneb_disk/data/stroke_inpainting/manifests")
    p.add_argument("--holdout-fraction", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=2026)
    args = p.parse_args()
    root = Path(args.root)
    rows = []
    suffix = "_desc-brainnorm_T1w.nii.gz"
    for t1 in sorted(root.glob("**/*" + suffix)):
        mask = t1.with_name(t1.name.replace(suffix, "_label-lesion_desc-T1lesion_mask.nii.gz"))
        if mask.is_file():
            site = t1.relative_to(root).parts[0]
            rows.append({"healthy_t1": str(t1), "exclude_mask": str(mask), "site": site})
    if len(rows) < 2:
        raise SystemExit(f"Found only {len(rows)} paired ATLAS3 files under {root}")
    sites = sorted({r["site"] for r in rows})
    rng = random.Random(args.seed); rng.shuffle(sites)
    n_holdout = max(1, min(len(sites)-1, round(len(sites) * args.holdout_fraction)))
    heldout_sites = set(sites[:n_holdout])
    train = [r for r in rows if r["site"] not in heldout_sites]
    heldout = [r for r in rows if r["site"] in heldout_sites]
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    for name, records in (("atlas3_train.csv", train), ("atlas3_holdout.csv", heldout)):
        with (output / name).open("w", newline="", encoding="utf8") as f:
            writer = csv.DictWriter(f, fieldnames=("healthy_t1", "exclude_mask", "site"))
            writer.writeheader(); writer.writerows(records)
    # Only masks from training sites contribute donor shapes, preventing site holdout leakage.
    with (output / "atlas3_masks_train.csv").open("w", newline="", encoding="utf8") as f:
        writer = csv.writer(f); writer.writerow(["lesion_mask"])
        writer.writerows((r["exclude_mask"],) for r in train)
    print(f"paired_scans={len(rows)} train={len(train)} holdout={len(heldout)} heldout_sites={n_holdout}")
    print(f"manifests={output}")

if __name__ == "__main__": main()
