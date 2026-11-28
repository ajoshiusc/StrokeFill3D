"""Matched-seed comparison of checkpoint and RF sampler choices."""

import argparse
import json
from pathlib import Path

import torch

from core import make_model, masked_metrics, sample
from train import dev_coords, load_templates, real_eval_masks


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--development", type=Path, required=True)
    p.add_argument("--heldout-mask", type=Path, action="append", required=True)
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--tuned", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.set_num_threads(3)
    device = torch.device("cuda")
    dev = torch.load(args.development, map_location="cpu", weights_only=True)
    dev = {k: v.to(device) for k, v in dev.items() if torch.is_tensor(v)}
    templates = load_templates(args.heldout_mask, min_area=130, max_area=1800)
    dev["mask"] = real_eval_masks(templates, dev["mask"])
    dev["image"] = torch.where(dev["mask"] > 0, 0, dev["target"])
    coords = dev_coords(dev)

    base_ckpt = torch.load(args.baseline, map_location="cpu", weights_only=True)
    tuned_ckpt = torch.load(args.tuned, map_location="cpu", weights_only=True)
    models = {}
    for name, ckpt in [("ellipse_base", base_ckpt), ("real_mask_tuned", tuned_ckpt)]:
        model = make_model(ckpt["model_config"]).to(device).eval()
        model.load_state_dict(ckpt["model"])
        models[name] = model

    configs = [
        ("base_legacy_euler20", "ellipse_base", "legacy_euler", 20),
        ("base_euler_endpoint20", "ellipse_base", "euler_endpoint", 20),
        ("base_heun_endpoint20", "ellipse_base", "heun_endpoint", 20),
        ("tuned_legacy_euler20", "real_mask_tuned", "legacy_euler", 20),
        ("tuned_euler_endpoint20", "real_mask_tuned", "euler_endpoint", 20),
        ("tuned_heun_endpoint20", "real_mask_tuned", "heun_endpoint", 20),
        ("tuned_heun_endpoint40", "real_mask_tuned", "heun_endpoint", 40),
    ]
    all_metrics, seed_outputs = {}, {}
    for label, model_key, sampler, steps in configs:
        rows, outputs = [], {}
        for seed in [909, 910]:
            pred = sample(
                models[model_key],
                dev["image"],
                dev["mask"],
                coords,
                seed,
                steps,
                sampler,
            )
            rows.append(masked_metrics(pred, dev["target"], dev["mask"]))
            if seed == 909:
                outputs[label] = pred
        all_metrics[label] = {
            key: sum(r[key] for r in rows) / len(rows) for key in rows[0]
        }
        if label == "tuned_heun_endpoint40":
            seed_outputs = outputs

    report = {
        "evaluation": "four clean held-out ATLAS patches; hard-case binary mask contours; same seeds 909 and 910",
        "metrics_are_mean_over_seeds": True,
        "methods": all_metrics,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "sampler_comparison.json").write_text(json.dumps(report, indent=2))
    # Show the sample-909 methods in one panel. This is a qualitative view;
    # the quantitative report always averages both fixed noise seeds.
    from PIL import Image, ImageDraw, ImageFont
    import numpy as np

    labels = [
        "Complete target",
        "Hidden input",
        "Base Euler20",
        "Tuned Euler20",
        "Tuned Heun20",
        "Tuned Heun40",
    ]
    pred_keys = [
        "base_legacy_euler20",
        "tuned_legacy_euler20",
        "tuned_heun_endpoint20",
        "tuned_heun_endpoint40",
    ]
    tile, gap, top, rowh = 192, 8, 48, 205
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
    canvas = Image.new(
        "RGB",
        (8 + len(labels) * (tile + gap), top + len(dev["target"]) * rowh),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (8, 8),
        "2D real-mask RF | held-out irregular contours | same seed 909",
        font=font,
        fill="black",
    )
    for j, label in enumerate(labels):
        draw.text((8 + j * (tile + gap), 28), label, font=font, fill="black")
    arrays = [dev["target"], dev["image"]] + [
        sample(
            models[model_key], dev["image"], dev["mask"], coords, 909, steps, sampler
        )
        for _, model_key, sampler, steps in [
            configs[0],
            configs[3],
            configs[5],
            configs[6],
        ]
    ]
    for i in range(len(dev["target"])):
        y = top + i * rowh
        for j, arr in enumerate(arrays):
            a = arr[i, 0].detach().cpu().numpy()
            if j == 1:
                a = np.where(dev["mask"][i, 0].cpu().numpy() > 0, -1, a)
            gray = np.uint8(np.clip((np.flipud(a.T) + 1) * 127.5, 0, 255))
            canvas.paste(
                Image.fromarray(gray).resize((tile, tile), Image.Resampling.NEAREST),
                (8 + j * (tile + gap), y + 5),
            )
    canvas.save(args.output / "sampler_comparison.png")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
