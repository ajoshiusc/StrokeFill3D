"""Split existing holdout sites into validation and reserved test manifests."""

import argparse
import csv
import random
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--holdout",
        default="/deneb_disk/data/stroke_inpainting/manifests/atlas3_holdout.csv",
    )
    p.add_argument(
        "--output-dir",
        default="/deneb_disk/data/stroke_inpainting/manifests/anatomical_v1",
    )
    p.add_argument("--seed", type=int, default=1729)
    args = p.parse_args()
    with open(args.holdout, newline="") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames
        rows = list(reader)
    sites = sorted({r["site"] for r in rows})
    if len(sites) < 2:
        raise ValueError("Need at least two holdout sites")
    random.Random(args.seed).shuffle(sites)
    val = set(sites[: max(1, len(sites) // 2)])
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, selected in [
        ("atlas3_val.csv", [r for r in rows if r["site"] in val]),
        ("atlas3_test.csv", [r for r in rows if r["site"] not in val]),
    ]:
        path = out / name
        if path.exists():
            with path.open(newline="") as f:
                existing = list(csv.DictReader(f))
            if existing != selected:
                raise ValueError(f"Refusing to overwrite a different split: {path}")
        else:
            with path.open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fields)
                w.writeheader()
                w.writerows(selected)
        print(
            f"{path}: {len(selected)} scans, {len({r['site'] for r in selected})} sites"
        )


if __name__ == "__main__":
    main()
