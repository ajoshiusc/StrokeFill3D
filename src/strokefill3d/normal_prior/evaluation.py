from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from strokefill3d.validation import reconstruction_metrics

from .core import coordinate_maps, fit_source
from .io import atomic_json


def metrics(pred, target, mask):
    if not torch.isfinite(pred).all():
        raise RuntimeError("Nonfinite prediction")
    return reconstruction_metrics(pred.float(), target.float(), mask.float())


def preview(path, case, pred, title):
    target = case["target"][0].cpu().numpy()
    image = case["image"][0].cpu().numpy()
    mask = case["mask"][0].cpu().numpy() > 0
    p = pred[0, 0].detach().cpu().numpy()
    center = np.argwhere(mask).mean(0).astype(int)
    font = ImageFont.load_default()
    size = 224
    canvas = Image.new("RGB", (size * 3, 60 + size * 3), (18, 22, 27))
    d = ImageDraw.Draw(canvas)
    d.text((8, 8), title, font=font, fill="white")
    for j, label in enumerate(
        ["Masked input", "Original normal target", "Generated RF / reconstruction"]
    ):
        d.text((8 + j * size, 35), label, font=font, fill="white")
    for row, axis in enumerate([1, 0, 2]):
        for col, a in enumerate([image, target, p]):
            tile = np.take(a, int(center[axis]), axis=axis).T
            if axis != 2:
                tile = np.flipud(tile)
            tile = Image.fromarray(
                np.uint8(np.clip((tile + 1) * 127.5, 0, 255))
            ).resize((size, size), Image.Resampling.NEAREST)
            canvas.paste(tile, (col * size, 60 + row * size))
    canvas.save(path)


def save_niftis(out, case, pred):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    affine = np.eye(4)
    affine[:3, 3] = case["origin"].numpy()
    for name, a in [
        ("target", case["target"]),
        ("masked_input", case["image"]),
        ("mask", case["mask"]),
        ("prediction", pred[0].detach().cpu()),
    ]:
        nib.save(
            nib.Nifti1Image(a[0].numpy().astype(np.float32), affine),
            out / f"{name}.nii.gz",
        )


@torch.inference_mode()
def evaluate_codec(codec_or_ae, cases, device, out=None, codec=False):
    rows = []
    for i, case in enumerate(cases):
        target = case["target"][None].to(device)
        mask = case["mask"][None].to(device)
        if codec:
            pred = codec_or_ae.decode(codec_or_ae.encode(target)).clamp(-1, 1)
        else:
            from .core import amp

            with amp(device):
                mu, _ = codec_or_ae.encode(target)
                pred = codec_or_ae.decode(mu).float().clamp(-1, 1)
        row = metrics(pred, target, mask)
        fg = metrics(pred, target, (target > -0.95).float())
        row["foreground_ssim"] = fg["masked_ssim"]
        rows.append(row)
        if out:
            Path(out).mkdir(parents=True, exist_ok=True)
            preview(
                Path(out) / f"oracle_{i:02d}.png",
                case,
                pred,
                "AE full-target reconstruction diagnostic: NOT inpainting",
            )
    return {
        "masked_ssim": float(np.mean([r["masked_ssim"] for r in rows])),
        "foreground_ssim": float(np.mean([r["foreground_ssim"] for r in rows])),
        "min_case_ssim": min(r["masked_ssim"] for r in rows),
        "masked_mae": float(np.mean([r["masked_mae"] for r in rows])),
        "cases": rows,
    }


def quality_passed(result, cfg):
    return (
        result["masked_ssim"] >= cfg["min_masked_ssim"]
        and result["foreground_ssim"] >= cfg["min_foreground_ssim"]
        and result["min_case_ssim"] >= cfg["min_case_ssim"]
    )


def evaluate_prior(model, codec, cases, device, out, settings, step):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in settings["seeds"]:
        for i, case in enumerate(cases[: settings["cases"]]):
            image = case["image"][None].to(device)
            mask = case["mask"][None].to(device)
            target = case["target"][None].to(device)
            coords = coordinate_maps(
                case["origin"][None].to(device),
                [n // codec.factor for n in codec.patch_size],
                codec.factor,
            )
            pred, _, _, history = fit_source(
                model,
                codec,
                image,
                mask,
                coords,
                seed=seed,
                steps=settings["flow_steps"],
                iterations=settings["iterations"],
                trust_weight=settings["trust_weight"],
                boundary_weight=settings["boundary_weight"],
                checkpoint_steps=True,
            )
            row = metrics(pred, target, mask)
            if row["outside_max_error"] != 0:
                raise RuntimeError("Observed voxels changed")
            row.update(
                case=i, seed=seed, source=case["source"], source_fit_history=history
            )
            rows.append(row)
            stem = out / f"case_{i:02d}_seed{seed}"
            preview(
                str(stem) + ".png",
                case,
                pred,
                f"Actual 3D normal-prior RF inpainting; step {step}; seed {seed}",
            )
            save_niftis(stem, case, pred)
    result = {
        "step": step,
        "protocol": settings,
        "masked_mae": float(np.mean([r["masked_mae"] for r in rows])),
        "masked_ssim": float(np.mean([r["masked_ssim"] for r in rows])),
        "gradient_mae": float(np.mean([r["gradient_mae"] for r in rows])),
        "cases": rows,
    }
    atomic_json(result, out / "metrics.json")
    return result
