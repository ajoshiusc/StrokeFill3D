"""Write a local CSV of ATLAS3 preprocessed T1 lesion masks."""
from __future__ import annotations
import argparse
import csv
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/deneb_disk/ATLAS3/ATLAS3_Training_Preprocessed")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.root)
    masks = sorted(root.glob("**/*_label-lesion_desc-T1lesion_mask.nii.gz"))
    if not masks:
        raise SystemExit(f"No preprocessed ATLAS3 lesion masks found under {root}")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf8") as f:
        writer = csv.writer(f)
        writer.writerow(["lesion_mask"])
        writer.writerows((str(p),) for p in masks)
    print(f"Wrote {len(masks)} mask paths to {output}")

if __name__ == "__main__":
    main()
