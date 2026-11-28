"""Fine-tune 2D conditional RF on clean targets and real lesion contours.

Lesion masks are read only as binary geometric templates. Their MR intensities
are never loaded. Target patches come from the existing lesion-excluded ATLAS3
training bank and retain the complete normal-appearing T1 target.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

from core import coords_from_starts, field, make_model, masked_metrics, sample


def cli():
    p = argparse.ArgumentParser()
    p.add_argument("--pairs", type=Path, required=True)
    p.add_argument("--initial", type=Path, required=True)
    p.add_argument("--development", type=Path, required=True)
    p.add_argument("--train-mask", type=Path, action="append", required=True)
    p.add_argument("--heldout-mask", type=Path, action="append", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--seed", type=int, default=271127)
    return p.parse_args()


def load_templates(paths, min_area=130, max_area=1450, max_extent=60):
    templates = []
    for path in paths:
        volume = np.asarray(nib.load(str(path)).dataobj) > 0
        if volume.shape != (197, 233, 189):
            raise ValueError(f"Expected canonical MNI mask grid: {path}")
        for y in range(54, 191):
            plane = volume[:, y, :]
            labels, n = ndimage.label(plane, structure=np.ones((3, 3), np.uint8))
            for label in range(1, n + 1):
                pts = np.argwhere(labels == label)
                area = len(pts)
                if not min_area <= area <= max_area:
                    continue
                lo, hi = pts.min(0), pts.max(0) + 1
                if np.max(hi - lo) > max_extent:
                    continue
                crop = labels[lo[0] : hi[0], lo[1] : hi[1]] == label
                templates.append(crop.astype(np.uint8))
    if len(templates) < 20:
        raise RuntimeError(f"Only found {len(templates)} usable contours")
    return templates


def make_mask_bank(templates, seed, variants_per_template=4):
    rng = np.random.default_rng(seed)
    bank = []
    for src in templates:
        for _ in range(variants_per_template):
            mask = src
            if rng.random() < 0.5:
                mask = mask[::-1, :]
            if rng.random() < 0.5:
                mask = mask[:, ::-1]
            angle = float(rng.uniform(-12, 12))
            scale = float(rng.uniform(0.9, 1.1))
            mask = ndimage.zoom(mask.astype(np.float32), scale, order=0) > 0.5
            mask = (
                ndimage.rotate(mask, angle, reshape=True, order=0, mode="constant")
                > 0.5
            )
            points = np.argwhere(mask)
            if not len(points):
                continue
            lo, hi = points.min(0), points.max(0) + 1
            crop = mask[lo[0] : hi[0], lo[1] : hi[1]]
            if max(crop.shape) > 60 or crop.sum() < 100:
                continue
            centered = np.zeros((64, 64), np.uint8)
            x0, z0 = (64 - crop.shape[0]) // 2, (64 - crop.shape[1]) // 2
            centered[x0 : x0 + crop.shape[0], z0 : z0 + crop.shape[1]] = crop
            bank.append(centered)
    if len(bank) < 100:
        raise RuntimeError(f"Only generated {len(bank)} mask variants")
    return torch.from_numpy(np.stack(bank)[:, None].copy()).float()


def donor_component_summary(templates):
    areas = np.array([m.sum() for m in templates])
    return {
        "count": int(len(areas)),
        "area_percentiles": np.percentile(areas, [0, 10, 25, 50, 75, 90, 100])
        .round(1)
        .tolist(),
    }


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def apply_center(mask, anchor):
    # Translate centered templates onto the old lesion-focus center, preserving
    # the learned cortical focus without retaining any ellipse geometry.
    n = len(mask)
    axis = torch.arange(64, device=mask.device).view(1, 64, 1)
    zaxis = torch.arange(64, device=mask.device).view(1, 1, 64)
    mass = anchor.sum((2, 3)).clamp_min(1)
    cx = (anchor[:, 0] * axis).sum((1, 2)) / mass[:, 0]
    cz = (anchor[:, 0] * zaxis).sum((1, 2)) / mass[:, 0]
    shiftx = (cx.round().long() - 32).view(n, 1, 1)
    shiftz = (cz.round().long() - 32).view(n, 1, 1)
    gridx = (torch.arange(64, device=mask.device).view(1, 64, 1) - shiftx).expand(
        -1, -1, 64
    )
    gridz = (torch.arange(64, device=mask.device).view(1, 1, 64) - shiftz).expand(
        -1, 64, -1
    )
    valid = (gridx >= 0) & (gridx < 64) & (gridz >= 0) & (gridz < 64)
    gridx = gridx.clamp(0, 63)
    gridz = gridz.clamp(0, 63)
    return (
        mask[torch.arange(n, device=mask.device)[:, None, None], 0, gridx, gridz][
            :, None
        ]
        * valid[:, None]
    )


def choose_batch(images, anchors, mask_bank, batch, device):
    # Resample until the synthetic lesion removes mostly brain tissue with
    # visible GM and WM, matching the cortical-focused subset of targets.
    chosen = []
    attempts = 0
    while len(chosen) < batch:
        attempts += 1
        if attempts > 50:
            raise RuntimeError(
                "Could not match real masks to complete cortical targets"
            )
        count = max(batch * 8, 64)
        ids = torch.randint(len(images), (count,), device=device)
        mids = torch.randint(len(mask_bank), (count,), device=device)
        target = images[ids].float()
        anchor = anchors[ids]
        masks = apply_center(mask_bank[mids], anchor)
        nvox = masks.flatten(1).sum(1).clamp_min(1)
        fg = ((target > -0.95) * masks).flatten(1).sum(1) / nvox
        wm = ((target > 0.5) * masks).flatten(1).sum(1) / nvox
        gm = (((target > -0.5) & (target < 0.4)) * masks).flatten(1).sum(1) / nvox
        valid = (fg > 0.85) & (wm > 0.06) & (gm > 0.06) & (nvox > 100) & (nvox < 1800)
        for j in valid.nonzero().flatten().tolist():
            chosen.append((ids[j], masks[j]))
            if len(chosen) == batch:
                break
    ix = torch.stack([v[0] for v in chosen])
    masks = torch.stack([v[1] for v in chosen])
    return ix, masks


def dev_coords(dev):
    c = dev["coords"]
    starts = torch.stack(
        [
            c[:, 0, 0, 0] * 100 + 98 - 1.5,
            c[:, 1, 0, 0] * 100,
            c[:, 2, 0, 0] * 100 + 72 - 1.5,
        ],
        1,
    )
    return coords_from_starts(starts)


def real_eval_masks(templates, dev_mask):
    # Held-out contours keep their extracted area and shape. Their centre is
    # anchored to each fixed validation focus so methods see identical masks.
    sorted_templates = sorted(templates, key=lambda x: int(x.sum()))
    masks = []
    areas = dev_mask.flatten(1).sum(1).cpu().numpy()
    for i, area in enumerate(areas):
        # Select a real contour near the validation mask's size; no target MRI
        # values participate in this selection.
        k = min(
            range(len(sorted_templates)),
            key=lambda j: abs(np.log((sorted_templates[j].sum() + 1) / (area + 1))),
        )
        src = sorted_templates[k]
        points = np.argwhere(src)
        crop = src[
            points.min(0)[0] : points.max(0)[0] + 1,
            points.min(0)[1] : points.max(0)[1] + 1,
        ]
        centered = np.zeros((64, 64), np.float32)
        x0, z0 = (64 - crop.shape[0]) // 2, (64 - crop.shape[1]) // 2
        centered[x0 : x0 + crop.shape[0], z0 : z0 + crop.shape[1]] = crop
        masks.append(torch.from_numpy(centered)[None])
    return apply_center(torch.stack(masks).to(dev_mask.device), dev_mask)


def render(path, target, image, mask, preds, title):
    labels = ["Complete target", "Hidden input", *preds.keys()]
    tile, gap, top, rowh = 192, 8, 44, 205
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
    canvas = Image.new(
        "RGB", (8 + len(labels) * (tile + gap), top + len(target) * rowh), "white"
    )
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 8), title, font=font, fill="black")
    for j, label in enumerate(labels):
        draw.text((8 + j * (tile + gap), 25), label, font=font, fill="black")
    arrays = [target, image, *preds.values()]
    for i in range(len(target)):
        y = top + i * rowh
        draw.text((8, y), f"Held-out mask {i + 1}", font=font, fill="black")
        for j, arr in enumerate(arrays):
            a = arr[i, 0].detach().cpu().numpy()
            if j == 1:
                m = mask[i, 0].detach().cpu().numpy() > 0
                a = np.where(m, -1, a)
            gray = np.uint8(np.clip((np.flipud(a.T) + 1) * 127.5, 0, 255))
            canvas.paste(
                Image.fromarray(gray).resize((tile, tile), Image.Resampling.NEAREST),
                (8 + j * (tile + gap), y + 18),
            )
    canvas.save(path)


def main():
    args = cli()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this bounded local experiment")
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(3)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda")
    tic = time.time()

    pairs = torch.load(args.pairs, map_location="cpu", weights_only=True)
    if pairs["metadata"].get("real_lesion_exclusion_margin_3d") != 3:
        raise ValueError("Target patches are not confirmed lesion-excluded")
    images = pairs["target"].to(device, non_blocking=True).float()
    anchors = pairs["mask"].to(device, non_blocking=True).float()
    starts = pairs["starts"].to(device, non_blocking=True)
    coords_all = coords_from_starts(starts)
    pair_sha = hashlib.sha256(args.pairs.read_bytes()).hexdigest()

    train_templates = load_templates(args.train_mask)
    heldout_templates = load_templates(args.heldout_mask, min_area=130, max_area=1800)
    mask_bank = make_mask_bank(train_templates, args.seed).to(device)
    dev = torch.load(args.development, map_location="cpu", weights_only=True)
    dev = {k: v.to(device) for k, v in dev.items() if torch.is_tensor(v)}
    dev["mask"] = real_eval_masks(heldout_templates, dev["mask"])
    dev["image"] = torch.where(dev["mask"] > 0, 0, dev["target"])
    dev["coords_full"] = dev_coords(dev)

    ckpt = torch.load(args.initial, map_location="cpu", weights_only=True)
    config = ckpt["model_config"]
    model = make_model(config).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    baseline = copy.deepcopy(model).eval().requires_grad_(False)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=5e-5, weight_decay=1e-4, fused=True
    )
    scaler = torch.amp.GradScaler("cuda", init_scale=1024)
    updates = 0

    protocol = {
        "experiment": "2D conditional rectified flow fine-tuned with held-out real lesion-mask geometry",
        "target_source": "complete ATLAS3 training T1 patches; 3D lesion exclusion margin = 3 voxels",
        "target_pairs_sha256": pair_sha,
        "initial_checkpoint_sha256": sha256(args.initial),
        "initial_checkpoint_step": ckpt.get("step"),
        "training_mask_sha256": [sha256(p) for p in args.train_mask],
        "heldout_mask_sha256": [sha256(p) for p in args.heldout_mask],
        "training_mask_geometry": donor_component_summary(train_templates),
        "heldout_mask_geometry": donor_component_summary(heldout_templates),
        "mask_augmentation": "binary flips, rotation ±12 degrees, scale 0.9–1.1",
        "optimizer": "fresh AdamW; initial optimizer state unavailable",
        "updates": args.steps,
        "batch_size": args.batch,
        "seed": args.seed,
        "objective": "masked RF velocity MSE; full clean target retained",
        "device": torch.cuda.get_device_name(0),
    }
    (args.output / "protocol.json").write_text(json.dumps(protocol, indent=2))

    @torch.no_grad()
    def evaluate(step):
        nonlocal model
        model.eval()
        metrics, outputs = {}, {}
        for label, net, method in [
            ("ellipse_base_heun_endpoint", baseline, "heun_endpoint"),
            ("real_mask_tuned_heun_endpoint", ema, "heun_endpoint"),
            ("ellipse_base_legacy_euler20", baseline, "legacy_euler"),
        ]:
            pred = sample(
                net, dev["image"], dev["mask"], dev["coords_full"], 909, 20, method
            )
            metrics[label] = masked_metrics(pred, dev["target"], dev["mask"])
            outputs[label] = pred
        report = {"step": step, "heldout_real_mask_metrics": metrics}
        (args.output / "history.json").write_text(json.dumps(report, indent=2))
        render(
            args.output / f"comparison_{step:05d}.png",
            dev["target"],
            dev["image"],
            dev["mask"],
            outputs,
            f"2D rectified flow | update {step} | identical noise seed 909 | held-out real contours",
        )
        print("EVAL", json.dumps(report), flush=True)
        model.train()

    initial_run = evaluate(0)
    history = []
    print(
        "TRAIN_START",
        json.dumps(
            {
                "steps": args.steps,
                "batch": args.batch,
                "device": torch.cuda.get_device_name(0),
                "target_patches": len(images),
                "lesion_exclusion_margin_3d": 3,
                "train_real_contours": donor_component_summary(train_templates),
                "heldout_real_contours": donor_component_summary(heldout_templates),
                "mask_variants": len(mask_bank),
                "checkpoint": str(args.initial),
                "baseline_checkpoint_step": ckpt.get("step"),
                "pairs_sha256": pair_sha,
                "objective": "RF masked velocity MSE; complete lesion-excluded T1 target; binary real contour only",
                "initial_optimizer_restored": False,
            }
        ),
        flush=True,
    )

    for step in range(1, args.steps + 1):
        ix, mask = choose_batch(images, anchors, mask_bank, args.batch, device)
        target = images[ix].float()
        coords = coords_all[ix]
        gamma = 0.9 + torch.rand(len(target), 1, 1, 1, device=device) * 0.2
        gain = 0.95 + torch.rand(len(target), 1, 1, 1, device=device) * 0.1
        target = torch.where(
            target > -0.95,
            (((target + 1) * 0.5).clamp_min(1e-5).pow(gamma) * gain).clamp(0, 1) * 2
            - 1,
            target,
        )
        image = torch.where(mask > 0, 0, target)
        eps = torch.randn_like(target)
        t = torch.rand(len(target), device=device) * 0.95
        tv = t[:, None, None, None]
        state = torch.where(mask > 0, (1 - tv) * eps + tv * target, image)
        optimizer.zero_grad(set_to_none=True)
        for g in optimizer.param_groups:
            g["lr"] = (
                5e-5
                * min(1, step / 100)
                * (0.2 + 0.8 * 0.5 * (1 + math.cos(math.pi * step / args.steps)))
            )
        model.train()
        velocity = field(model, state, image, mask, coords, t)
        loss = (
            ((velocity - target + eps).square() * mask).flatten(1).sum(1)
            / mask.flatten(1).sum(1).clamp_min(1)
        ).mean()
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite RF loss")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        old_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        if scaler.get_scale() >= old_scale:
            updates += 1
            with torch.no_grad():
                torch._foreach_lerp_(
                    list(ema.parameters()),
                    list(model.parameters()),
                    1 - min(0.999, (updates + 1) / (updates + 10)),
                )
        if step % 100 == 0:
            print(
                "STEP",
                step,
                "loss",
                float(loss.detach()),
                "minutes",
                round((time.time() - tic) / 60, 2),
                flush=True,
            )
        if step % 1000 == 0 or step == args.steps:
            evaluate(step)
            state_out = {
                "kind": "real_mask_conditional_rf_2d_v1",
                "step": step,
                "model_config": config,
                "model": {k: v.detach().cpu() for k, v in ema.state_dict().items()},
                "metrics": json.loads((args.output / "history.json").read_text()),
                "training_pairs_sha256": pair_sha,
            }
            torch.save(state_out, args.output / f"real_mask_rf_step{step:05d}.pt")
            torch.save(
                {
                    "model": {
                        k: v.detach().cpu() for k, v in model.state_dict().items()
                    },
                    "ema": {k: v.detach().cpu() for k, v in ema.state_dict().items()},
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict(),
                    "step": step,
                    "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state(),
                    "training_pairs_sha256": pair_sha,
                },
                args.output / "training_state.pt",
            )
    print("FINISHED", round((time.time() - tic) / 60, 2), "minutes", flush=True)


if __name__ == "__main__":
    main()
