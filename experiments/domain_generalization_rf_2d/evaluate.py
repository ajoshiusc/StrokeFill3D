"""Evaluate domain-mixed RF variants on fixed clean targets and real contours."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).parents[1] / "real_mask_rf_2d"))
from core import coords_from_starts, make_model, masked_metrics, sample
from train import real_eval_masks


def load_model(path, device):
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    model = make_model(ckpt["model_config"]).to(device).eval()
    model.load_state_dict(ckpt["model"])
    return model


def evaluate_cohort(models, target, mask, starts, domains, device, batch=16):
    target = target.to(device).float()
    mask = mask.to(device).float()
    starts = starts.to(device).float()
    coords = coords_from_starts(starts)
    image = torch.where(mask > 0, 0, target)
    sums = {m: {} for m in models}
    counts = {m: {} for m in models}
    examples = {}
    for replicate, seed_offset in enumerate([0, 10000]):
        for i in range(0, len(target), batch):
            sl = slice(i, min(i + batch, len(target)))
            domain_chunk = domains[i : sl.stop]
            d = target[sl]
            m = mask[sl]
            inp, c = image[sl], coords[sl]
            seed = 909 + seed_offset + i // batch
            for name, model in models.items():
                pred = sample(model, inp, m, c, seed, 20, "legacy_euler")
                rows = [
                    masked_metrics(pred[j : j + 1], d[j : j + 1], m[j : j + 1])
                    for j in range(len(d))
                ]
                for j, domain in enumerate(domain_chunk):
                    sums[name].setdefault(domain, {k: 0.0 for k in rows[j]})
                    counts[name][domain] = counts[name].get(domain, 0) + 1
                    for key, value in rows[j].items():
                        sums[name][domain][key] += value
                    if replicate == 0:
                        if domain not in examples:
                            examples[domain] = {
                                "index": i + j,
                                "target": d[j : j + 1].cpu(),
                                "input": inp[j : j + 1].cpu(),
                                "mask": m[j : j + 1].cpu(),
                                "predictions": {},
                            }
                        if name not in examples[domain]["predictions"]:
                            examples[domain]["predictions"][name] = pred[
                                j : j + 1
                            ].cpu()
    metrics = {}
    for model_name in models:
        metrics[model_name] = {}
        for domain in sums[model_name]:
            n = counts[model_name][domain]
            metrics[model_name][domain] = {
                k: v / n for k, v in sums[model_name][domain].items()
            }
    return metrics, examples


def render(path, examples, model_names, title):
    domains = [d for d in ["episurg", "epibios", "arc"] if d in examples]
    labels = ["Complete target", "Hidden input"] + [
        m.replace("_", " ") for m in model_names
    ]
    tile, gap, top, rowh = 192, 8, 76, 224
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
    canvas = Image.new(
        "RGB", (8 + len(labels) * (tile + gap), top + len(domains) * rowh), "white"
    )
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 8), title, font=font, fill="black")
    for j, label in enumerate(labels):
        draw.text((8 + j * (tile + gap), 28), label, font=font, fill="black")
    for row, domain in enumerate(domains):
        item = examples[domain]
        arrays = [item["target"], item["input"]] + [
            item["predictions"][m] for m in model_names
        ]
        y = top + row * rowh
        draw.text(
            (8, y + 2), f"Held-out {domain.upper()} patch", font=font, fill="black"
        )
        for j, arr in enumerate(arrays):
            a = arr[0, 0].numpy()
            if j == 1:
                a = np.where(item["mask"][0, 0].numpy() > 0, -1, a)
            gray = np.uint8(np.clip((np.flipud(a.T) + 1) * 127.5, 0, 255))
            canvas.paste(
                Image.fromarray(gray).resize((tile, tile), Image.Resampling.NEAREST),
                (8 + j * (tile + gap), y + 22),
            )
    canvas.save(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mixed-data", type=Path, required=True)
    p.add_argument("--atlas-development", type=Path, required=True)
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--standard", type=Path, required=True)
    p.add_argument("--hist-bias", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    torch.set_num_threads(3)
    device = torch.device("cuda")
    models = {
        "atlas_only": load_model(args.baseline, device),
        "mixed_standard": load_model(args.standard, device),
        "mixed_hist_bias": load_model(args.hist_bias, device),
    }
    data = torch.load(args.mixed_data, map_location="cpu", weights_only=True)
    ext = data["test"]
    ext_metrics, ext_examples = evaluate_cohort(
        models, ext["target"], ext["mask"], ext["starts"], ext["domain"], device
    )
    dev = torch.load(args.atlas_development, map_location="cpu", weights_only=True)
    templates = data["heldout_mask_templates"][:, 0].numpy().astype(np.uint8)
    atlas_masks = real_eval_masks(list(templates), dev["mask"])
    atlas_metrics, atlas_examples = evaluate_cohort(
        models,
        dev["target"],
        atlas_masks.cpu(),
        torch.stack(
            [
                dev["coords"][:, 0, 0, 0] * 100 + 98 - 1.5,
                dev["coords"][:, 1, 0, 0] * 100,
                dev["coords"][:, 2, 0, 0] * 100 + 72 - 1.5,
            ],
            1,
        ),
        ["atlas"] * len(dev["target"]),
        device,
        batch=4,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    report = {
        "protocol": "paired synthetic inpainting; complete targets clear of a 5-mm 3D lesion margin; legacy Euler20; identical noise seeds 909 and 10909; held-out participants/cases",
        "external_test": ext_metrics,
        "atlas_retention": atlas_metrics,
        "data_split": data["metadata"],
    }
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2))
    render(
        args.output / "domain_comparison.png",
        ext_examples,
        ["atlas_only", "mixed_standard", "mixed_hist_bias"],
        "2D RF domain test | held-out clean targets and real lesion contours",
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
