"""Mixed-domain RF fine-tuning with optional histogram and bias-field changes."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parents[1] / "real_mask_rf_2d"))
from core import coords_from_starts, field, make_model
from train import make_mask_bank


def place_masks(templates, centers):
    n = len(templates)
    cx = centers[:, 0].round().long()
    cz = centers[:, 1].round().long()
    xx = torch.arange(64, device=templates.device).view(1, 64, 1)
    zz = torch.arange(64, device=templates.device).view(1, 1, 64)
    gx = xx - (cx - 32).view(n, 1, 1)
    gz = zz - (cz - 32).view(n, 1, 1)
    valid = (gx >= 0) & (gx < 64) & (gz >= 0) & (gz < 64)
    gx, gz = gx.clamp(0, 63), gz.clamp(0, 63)
    return (
        templates[torch.arange(n, device=templates.device)[:, None, None], 0, gx, gz][
            :, None
        ]
        * valid[:, None]
    )


def histogram_warp(target, generator):
    b = len(target)
    increments = 0.65 + 0.70 * torch.rand(
        (b, 6), device=target.device, generator=generator
    )
    knots = torch.cat(
        [torch.zeros((b, 1), device=target.device), increments.cumsum(1)], 1
    )
    knots = knots / knots[:, -1:]
    unit = ((target + 1) * 0.5).clamp(0, 1)
    pos = unit * 6
    lo = pos.floor().long().clamp(0, 5)
    frac = pos - lo
    y0 = torch.gather(knots, 1, lo.flatten(1)).view_as(target)
    y1 = torch.gather(knots, 1, (lo + 1).flatten(1)).view_as(target)
    warped = 2 * (y0 + frac * (y1 - y0)) - 1
    return torch.where(target > -0.95, warped, -1)


def augment(target, mode, generator):
    # Mild global contrast changes are used in both arms. The second arm adds
    # monotone histogram warps and a smooth multiplicative coil field.
    gamma = (
        0.9
        + torch.rand(len(target), 1, 1, 1, device=target.device, generator=generator)
        * 0.2
    )
    gain = (
        0.95
        + torch.rand(len(target), 1, 1, 1, device=target.device, generator=generator)
        * 0.1
    )
    target = torch.where(
        target > -0.95,
        (((target + 1) * 0.5).clamp_min(1e-5).pow(gamma) * gain).clamp(0, 1) * 2 - 1,
        target,
    )
    if mode == "hist_bias":
        target = histogram_warp(target, generator)
        lowres = torch.randn(
            (len(target), 1, 5, 5), device=target.device, generator=generator
        )
        bias = F.interpolate(lowres, size=(64, 64), mode="bicubic", align_corners=False)
        bias = bias / bias.std((2, 3), keepdim=True).clamp_min(1e-3)
        strength = 0.05 + 0.13 * torch.rand(
            (len(target), 1, 1, 1), device=target.device, generator=generator
        )
        multiplier = torch.exp(bias * strength)
        unit = ((target + 1) * 0.5 * multiplier).clamp(0, 1)
        target = torch.where(target > -0.95, 2 * unit - 1, -1)
    elif mode != "standard":
        raise ValueError(mode)
    return target


def choose_batch(
    targets, starts, centers, group_indices, mask_bank, batch, device, generator
):
    chosen_ids, chosen_masks = [], []
    attempts = 0
    domain_names = list(group_indices)
    domain_weights = torch.tensor(
        [3.0 if n == "atlas" else 1.0 for n in domain_names], device=device
    )
    while len(chosen_ids) < batch:
        attempts += 1
        if attempts > 60:
            raise RuntimeError(
                "Could not draw real masks covering clean cortical tissue"
            )
        count = max(batch * 8, 64)
        domains = torch.multinomial(
            domain_weights, count, replacement=True, generator=generator
        )
        ids = torch.empty(count, dtype=torch.long, device=device)
        for group_id, name in enumerate(domain_names):
            sel = domains == group_id
            pool = group_indices[name]
            ids[sel] = pool[
                torch.randint(
                    len(pool), (int(sel.sum()),), device=device, generator=generator
                )
            ]
        mids = torch.randint(
            len(mask_bank), (count,), device=device, generator=generator
        )
        masks = place_masks(mask_bank[mids], centers[ids])
        target = targets[ids].float()
        nvox = masks.flatten(1).sum(1).clamp_min(1)
        fg = ((target > -0.95) * masks).flatten(1).sum(1) / nvox
        wm = ((target > 0.5) * masks).flatten(1).sum(1) / nvox
        gm = (((target > -0.5) & (target < 0.4)) * masks).flatten(1).sum(1) / nvox
        ok = (fg >= 0.82) & (wm >= 0.04) & (gm >= 0.04) & (nvox >= 100) & (nvox <= 1800)
        for j in ok.nonzero().flatten().tolist():
            chosen_ids.append(ids[j])
            chosen_masks.append(masks[j])
            if len(chosen_ids) == batch:
                break
    ix = torch.stack(chosen_ids)
    return ix, torch.stack(chosen_masks), coords_from_starts(starts[ix])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--atlas-pairs", type=Path, required=True)
    p.add_argument("--mixed-data", type=Path, required=True)
    p.add_argument("--initial", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--mode", choices=["standard", "hist_bias"], required=True)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--seed", type=int, default=271129)
    args = p.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    torch.set_num_threads(3)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda")
    data_rng = torch.Generator(device=device).manual_seed(args.seed + 101)
    flow_rng = torch.Generator(device=device).manual_seed(args.seed + 202)
    augment_rng = torch.Generator(device=device).manual_seed(args.seed + 303)
    out = args.output
    out.mkdir(parents=True, exist_ok=True)

    atlas = torch.load(args.atlas_pairs, map_location="cpu", weights_only=True)
    mixed = torch.load(args.mixed_data, map_location="cpu", weights_only=True)
    if atlas["metadata"].get("real_lesion_exclusion_margin_3d") != 3:
        raise ValueError("ATLAS clean target lesion exclusion metadata missing")
    if not mixed["metadata"].get("targets_are_complete_clean_patches"):
        raise ValueError(
            "Mixed-domain patch bank is not marked as complete clean targets"
        )
    if mixed["metadata"].get("lesion_image_intensities_used_as_targets"):
        raise ValueError("Refusing a mixed-domain bank containing lesion targets")

    atlas_target = atlas["target"].to(device).float()
    atlas_starts = atlas["starts"].to(device).float()
    atlas_anchor = atlas["mask"].to(device).float()
    coords = torch.arange(64, device=device).view(1, 64, 1)
    zcoords = torch.arange(64, device=device).view(1, 1, 64)
    den = atlas_anchor.sum((2, 3)).clamp_min(1)
    atlas_centers = torch.stack(
        [
            (atlas_anchor[:, 0] * coords).sum((1, 2)) / den[:, 0],
            (atlas_anchor[:, 0] * zcoords).sum((1, 2)) / den[:, 0],
        ],
        1,
    )
    ext = mixed["train"]
    ext_target = ext["target"].to(device).float()
    ext_starts = ext["starts"].to(device).float()
    ext_centers = ext["centers"].to(device).float()
    targets = torch.cat([atlas_target, ext_target])
    starts = torch.cat([atlas_starts, ext_starts])
    centers = torch.cat([atlas_centers, ext_centers])
    offset = len(atlas_target)
    groups = {"atlas": torch.arange(offset, device=device)}
    for domain in ["episurg", "epibios", "arc"]:
        idx = [offset + i for i, d in enumerate(ext["domain"]) if d == domain]
        if idx:
            groups[domain] = torch.tensor(idx, device=device)
    if set(groups) != {"atlas", "episurg", "epibios", "arc"}:
        raise RuntimeError(
            f"Expected all four balanced target domains, got {list(groups)}"
        )
    source_masks = mixed["train_mask_templates"][:, 0].numpy().astype("uint8")
    mask_bank = make_mask_bank(list(source_masks), args.seed).to(device)

    initial = torch.load(args.initial, map_location="cpu", weights_only=True)
    model = make_model(initial["model_config"]).to(device)
    model.load_state_dict(initial["model"])
    ema = make_model(initial["model_config"]).to(device).eval().requires_grad_(False)
    ema.load_state_dict(initial["model"])
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=4e-5, weight_decay=1e-4, fused=True
    )
    scaler = torch.amp.GradScaler("cuda", init_scale=1024)
    tic = time.time()
    history = []
    protocol = {
        "mode": args.mode,
        "steps": args.steps,
        "batch": args.batch,
        "seed": args.seed,
        "initial_checkpoint_step": initial.get("step"),
        "target_mix": "50% ATLAS, 50% evenly sampled EPISURG/EPIBIOS/ARC",
        "lesion_exclusion_mm_external": 5,
        "loss": "RF velocity MSE inside real irregular mask",
        "augmentation": "gamma/gain"
        if args.mode == "standard"
        else "gamma/gain plus monotone histogram warp and smooth multiplicative bias field",
        "mixed_data_metadata": mixed["metadata"],
        "initial_checkpoint_sha256": hashlib.sha256(
            args.initial.read_bytes()
        ).hexdigest(),
        "mixed_data_sha256": hashlib.sha256(args.mixed_data.read_bytes()).hexdigest(),
        "device": torch.cuda.get_device_name(0),
    }
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2))

    for step in range(1, args.steps + 1):
        ix, mask, coord = choose_batch(
            targets, starts, centers, groups, mask_bank, args.batch, device, data_rng
        )
        target = augment(targets[ix].float(), args.mode, augment_rng)
        image = torch.where(mask > 0, 0, target)
        eps = torch.randn(
            target.shape, device=device, dtype=target.dtype, generator=flow_rng
        )
        t = torch.rand(len(target), device=device, generator=flow_rng) * 0.95
        tv = t[:, None, None, None]
        state = torch.where(mask > 0, (1 - tv) * eps + tv * target, image)
        for g in optimizer.param_groups:
            g["lr"] = (
                4e-5
                * min(1, step / 100)
                * (0.2 + 0.8 * 0.5 * (1 + math.cos(math.pi * step / args.steps)))
            )
        optimizer.zero_grad(set_to_none=True)
        model.train()
        velocity = field(model, state, image, mask, coord, t)
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
            with torch.no_grad():
                torch._foreach_lerp_(
                    list(ema.parameters()), list(model.parameters()), 0.01
                )
        if step % 100 == 0:
            history.append({"step": step, "loss": float(loss.detach())})
            print(
                "STEP",
                step,
                "loss",
                round(float(loss.detach()), 5),
                "minutes",
                round((time.time() - tic) / 60, 2),
                flush=True,
            )
        if step % 1000 == 0 or step == args.steps:
            torch.save(
                {
                    "kind": "mixed_domain_real_mask_rf_2d_v1",
                    "mode": args.mode,
                    "step": step,
                    "model_config": initial["model_config"],
                    "model": {k: v.detach().cpu() for k, v in ema.state_dict().items()},
                    "initial_checkpoint_sha256": protocol["initial_checkpoint_sha256"],
                    "mixed_data_sha256": protocol["mixed_data_sha256"],
                    "history": history,
                },
                out / f"model_step{step:05d}.pt",
            )
            (out / "history.json").write_text(json.dumps(history, indent=2))
    print(
        "FINISHED", args.mode, round((time.time() - tic) / 60, 2), "minutes", flush=True
    )


if __name__ == "__main__":
    main()
