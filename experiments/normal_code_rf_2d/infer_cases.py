"""Run the tested 2D normal-code RF on prepared, normalized MRI patches."""

from pathlib import Path
import argparse, hashlib, json, time
import numpy as np, torch
import torch.nn.functional as F
from monai.metrics.regression import compute_ssim_and_cs
from PIL import Image, ImageDraw, ImageFont
from core import load_run, fit_source, inpaint


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--checkpoint", default="last.pt")
    p.add_argument("--cases", type=Path, required=True)
    p.add_argument("--key")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=909)
    p.add_argument("--steps", type=int, default=16)
    p.add_argument("--iterations", type=int, default=120)
    p.add_argument("--trust", type=float, default=0.001)
    p.add_argument("--boundary-weight", type=float, default=4.0)
    p.add_argument("--method", choices=["source", "projection"], default="source")
    args = p.parse_args()
    torch.set_num_threads(3)
    start = time.time()
    model, codec, checkpoint = load_run(args.run, args.checkpoint, args.device)
    loaded = torch.load(args.cases, map_location="cpu", weights_only=True)
    if args.key:
        loaded = loaded[args.key]
    image = loaded["image"].float().to(args.device)
    mask = loaded["mask"].float().to(args.device)
    image = torch.where(mask > 0, torch.zeros_like(image), image)
    if image.min() < -1.001 or image.max() > 1.001:
        raise ValueError(
            "Expected full-volume percentile-normalized [-1,1] MRI patches"
        )
    if "coords" in loaded:
        coords = loaded["coords"].float().to(args.device)
    elif "anatomy" in loaded:
        coords = F.avg_pool2d(loaded["anatomy"][:, 2:5].float().to(args.device), 4)
    else:
        raise ValueError(
            "Provide coords (Bx3x16x16) or anatomy with MNI xyz in channels 2:5"
        )
    if args.method == "source":
        prediction, code, history, source = fit_source(
            model,
            codec,
            image,
            mask,
            coords,
            seed=args.seed,
            steps=args.steps,
            iterations=args.iterations,
            trust_weight=args.trust,
            boundary_weight=args.boundary_weight,
            return_source=True,
        )
    else:
        prediction, code = inpaint(
            model,
            codec,
            image,
            mask,
            coords,
            seed=args.seed,
            steps=args.steps,
            projection=checkpoint.get("projection", 1.0),
            return_code=True,
        )
        history = []
        source = None
    assert torch.isfinite(prediction).all() and torch.equal(
        prediction[mask == 0], image[mask == 0]
    )
    report = dict(
        checkpoint_step=checkpoint["step"],
        method=args.method,
        flow_steps=args.steps,
        source_iterations=args.iterations,
        trust_weight=args.trust,
        boundary_weight=args.boundary_weight,
        seed=args.seed,
        seconds=time.time() - start,
        target_used_for_inference=False,
        history=history,
    )
    for key, path in [
        ("checkpoint_sha256", args.run / args.checkpoint),
        ("codec_sha256", args.run / "codec.pt"),
    ]:
        report[key] = hashlib.sha256(path.read_bytes()).hexdigest()
    columns = [("Masked input", image), ("Normal-prior RF", prediction)]
    # Targets are accessed only after generation, solely for evaluation/display.
    if "target" in loaded:
        target = loaded["target"].float().to(args.device)
        den = mask.flatten(1).sum(1)
        if torch.any(den == 0):
            raise ValueError("An evaluation case has no hidden pixels")
        mm = mask[:, :, 3:-3, 3:-3]
        ss = compute_ssim_and_cs(
            prediction, target, 2, (7, 7), (1.5, 1.5), data_range=2
        )[0]
        report["masked_mae"] = float(
            (((prediction - target).abs() * mask).flatten(1).sum(1) / den).mean()
        )
        report["masked_ssim"] = float(
            ((ss * mm).flatten(1).sum(1) / mm.flatten(1).sum(1)).mean()
        )
        columns.insert(0, ("Original normal target", target))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        dict(
            prediction=prediction.cpu(),
            generated_code=code.cpu(),
            optimized_source=None if source is None else source.cpu(),
            report=report,
        ),
        args.output.with_suffix(".pt"),
    )
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2))
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    font = (
        ImageFont.truetype(font_path, 16)
        if Path(font_path).exists()
        else ImageFont.load_default()
    )
    for begin in range(0, len(image), 4):
        ids = list(range(begin, min(begin + 4, len(image))))
        size = 224
        gap = 12
        canvas = Image.new(
            "RGB",
            (len(columns) * (size + gap) + gap, 90 + len(ids) * 246),
            (18, 22, 27),
        )
        draw = ImageDraw.Draw(canvas)
        draw.text(
            (gap, 10),
            f"Actual RF inpainting, checkpoint {checkpoint['step']}; seed {args.seed}. Superior up.",
            font=font,
            fill="white",
        )
        for col, (label, _) in enumerate(columns):
            draw.text((gap + col * (size + gap), 45), label, font=font, fill="white")
        for row, i in enumerate(ids):
            for col, (_, a) in enumerate(columns):
                ar = np.flipud(a[i, 0].detach().cpu().numpy().T)
                tile = Image.fromarray(
                    np.uint8(np.clip((ar + 1) * 127.5, 0, 255))
                ).resize((size, size), Image.Resampling.NEAREST)
                canvas.paste(tile, (gap + col * (size + gap), 80 + row * 246))
        canvas.save(
            args.output.with_name(
                f"{args.output.stem}_{begin + 1:02d}_{begin + len(ids):02d}"
            ).with_suffix(".png")
        )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
