"""Freeze prepared external scans and build subject-disjoint mixed RF manifests."""

import argparse
import csv
import json
import shutil
from pathlib import Path

FIELDS = ["healthy_t1", "exclude_mask", "brain_mask", "subject", "domain", "site"]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--atlas-manifests", type=Path, required=True)
    p.add_argument(
        "--cases", type=Path, required=True, help="Prepared four-case directory"
    )
    p.add_argument("--arc-manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if (a.output / "train.csv").exists():
        raise ValueError("Dataset already frozen; use a new directory")
    a.output.mkdir(parents=True, exist_ok=True)
    groups = {}
    for split, name in [
        ("train", "atlas3_train.csv"),
        ("val", "anatomical_v1/atlas3_val.csv"),
        ("test", "anatomical_v1/atlas3_test.csv"),
    ]:
        rows = list(csv.DictReader((a.atlas_manifests / name).open(newline="")))
        groups[split] = [
            {
                "healthy_t1": r["healthy_t1"],
                "exclude_mask": r["exclude_mask"],
                "brain_mask": "",
                "subject": "atlas:" + Path(r["healthy_t1"]).parent.parent.parent.name,
                "domain": "atlas",
                "site": r.get("site", ""),
            }
            for r in rows
        ]

    def stage(t1, lesion, brain, subject, domain, split):
        dest = a.output / "volumes" / subject
        dest.mkdir(parents=True, exist_ok=True)
        paths = {}
        for key, src in [
            ("healthy_t1", t1),
            ("exclude_mask", lesion),
            ("brain_mask", brain),
        ]:
            src = Path(src)
            if not src.is_file():
                raise FileNotFoundError(src)
            dst = dest / (key + ".nii.gz")
            shutil.copy2(src, dst)
            paths[key] = str(dst.resolve())
        groups[split].append(dict(**paths, subject=subject, domain=domain, site=domain))

    for domain, prefix in [("episurg", "ep_s"), ("epibios", "epi_b")]:
        for level, split in [("medium", "train"), ("hard", "val")]:
            c = a.cases / (prefix + "_" + level)
            stage(
                c / "t1_brain_training.nii.gz",
                c / "lesion_training.nii.gz",
                c / "brain_mask_training.nii.gz",
                domain + "_" + level,
                domain,
                split,
            )
    # One scan per participant avoids overweighting longitudinal participants.
    rows = list(csv.DictReader(a.arc_manifest.open(newline="")))
    first = {}
    for r in rows:
        first.setdefault(r["subject"], r)
    subjects = sorted(first)
    if len(subjects) < 6:
        raise ValueError("Need >=6 ARC participants for a train/val/test split")
    for i, subject in enumerate(subjects):
        split = (
            "val"
            if i == len(subjects) - 2
            else "test"
            if i == len(subjects) - 1
            else "train"
        )
        r = first[subject]
        stage(
            r["brain_t1_mni"],
            r["lesion_mask_mni"],
            r["skullstrip_mask_mni"],
            "arc_" + subject,
            "arc",
            split,
        )
    for split, rows in groups.items():
        with (a.output / (split + ".csv")).open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(rows)
    metadata = {
        "counts": {k: len(v) for k, v in groups.items()},
        "external_mask_provenance": "EPISURG/EPIBIOS supplied labels; ARC stroke_inpainting pipeline nnUNet predictions (require visual review for missed lesions)",
        "exclusion_mm": 5,
        "notes": "Validation has all four domains. Test has ATLAS and separate ARC participant; more EPI participants needed for an independent EPI test set.",
    }
    (a.output / "dataset.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
