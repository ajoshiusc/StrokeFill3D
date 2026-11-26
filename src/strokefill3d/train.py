"""Conditional rectified flow with EMA and fixed held-out generation validation."""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import socket
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from .anatomy import ANATOMY_CHANNELS, ANATOMY_VERSION
from .losses import training_loss
from .model import RectifiedFlowUNet, model_config
from .validation import (
    check_split,
    dataset_from_config,
    evaluate,
    file_hash,
    fixed_cases,
)


def seed_worker(worker_id):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def atomic_save(state, path):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(state, tmp)
    tmp.replace(path)


def compatible_best_score(state):
    """Do not compare the previous bbox-based score with masked-SSIM scores."""
    metrics = state.get("metrics") or {}
    if state.get("validation_metric_version") == 2 or "masked_ssim" in metrics:
        return state.get("best_score", float("inf"))
    return float("inf")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/train.yaml")
    p.add_argument("--manifest", required=True)
    p.add_argument("--mask-manifest")
    p.add_argument("--val-manifest")
    p.add_argument("--test-manifest")
    p.add_argument("--output", default="checkpoints")
    p.add_argument("--resume")
    p.add_argument(
        "--preflight",
        action="store_true",
        help="One optimizer update and short generation check; no training checkpoints",
    )
    p.add_argument("--device", choices=["cuda", "cpu"])
    p.add_argument("--max-steps", type=int)
    args = p.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    if args.max_steps is not None:
        cfg["max_steps"] = args.max_steps
    seed = cfg.get("seed", 2026)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable")
    if cfg.get("anatomy_channels", 0) not in (0, ANATOMY_CHANNELS):
        raise ValueError("Unsupported anatomy channel count")
    if cfg.get("require_validation", False) and not args.val_manifest:
        p.error(
            "This profile requires --val-manifest (use scripts/split_validation_sites.py)"
        )
    if args.val_manifest:
        check_split(
            args.manifest, args.val_manifest, args.mask_manifest, args.test_manifest
        )
    out = Path(args.output)
    if not args.preflight and not args.resume and (out / "last.pt").exists():
        raise ValueError("Output contains a run. Use --resume or a new run directory")
    out.mkdir(parents=True, exist_ok=True)
    # Advisory process lock prevents concurrent writers even when data are shared across hosts.
    import fcntl

    lock = (out / ".train.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError("Another trainer holds this run directory lock")
    lock.seek(0)
    lock.truncate()
    lock.write(f"{socket.gethostname()}\n")
    lock.flush()
    identities = {
        k: file_hash(v)
        for k, v in {
            "train": args.manifest,
            "masks": args.mask_manifest,
            "validation": args.val_manifest,
            "test": args.test_manifest,
        }.items()
        if v
    }
    (out / "resolved_config.yaml").write_text(yaml.safe_dump(cfg))
    dataset = dataset_from_config(cfg, args.manifest, args.mask_manifest)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg["num_workers"],
        pin_memory=device.type == "cuda",
        worker_init_fn=seed_worker,
        generator=generator,
        persistent_workers=cfg["num_workers"] > 0,
    )
    mc = model_config(cfg)
    model = RectifiedFlowUNet(**mc).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"]
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and cfg["amp"])
    ema = copy.deepcopy(model).eval() if cfg.get("ema_decay", 0) else None
    if ema:
        for parameter in ema.parameters():
            parameter.requires_grad_(False)
    accumulation = int(cfg.get("gradient_accumulation", 1))
    if accumulation < 1:
        raise ValueError("gradient_accumulation must be positive")
    steps_per_epoch = math.ceil(len(loader) / accumulation)
    max_steps = int(cfg.get("max_steps", steps_per_epoch * cfg["epochs"]))
    if max_steps < 1:
        raise ValueError("max_steps must be positive")
    step = 0
    start_epoch = 0
    best = float("inf")
    last_val = -1
    last_metrics = None
    if args.resume:
        state = torch.load(args.resume, map_location="cpu", weights_only=True)
        if state["model_config"] != mc:
            raise ValueError(
                "Architecture differs. Start a fresh anatomical run; old checkpoints cannot be resumed into it"
            )
        if state.get("data_hashes", identities) != identities:
            raise ValueError("Resume manifests differ from checkpoint")
        if state.get("anatomy_version") != (
            ANATOMY_VERSION if mc.get("anatomy_channels", 0) else None
        ):
            raise ValueError("Checkpoint conditioning version differs")
        critical = (
            "patch_size",
            "spacing_mm",
            "exclusion_margin",
            "partial_mask_probability",
            "large_mask_probability",
            "max_mask_dilation_voxels",
            "endpoint_l1_weight",
            "gradient_weight",
            "structural_min_time",
            "gradient_accumulation",
        )
        if mc.get("anatomy_channels", 0) and any(
            state.get("training_config", {}).get(k) != cfg.get(k) for k in critical
        ):
            raise ValueError(
                "Resume data/objective settings differ; start a separate run"
            )
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        if state.get("scaler"):
            scaler.load_state_dict(state["scaler"])
        if ema:
            ema.load_state_dict(state.get("ema_model") or state["model"])
        step = int(state["step"])
        start_epoch = int(state["epoch"]) - (not state.get("epoch_completed", False))
        best = compatible_best_score(state)
        last_metrics = state.get("metrics") if best != float("inf") else None
        if step >= max_steps:
            raise ValueError(
                "Checkpoint already reached max_steps; increase it to extend the run"
            )
    cases = (
        fixed_cases(cfg, args.val_manifest, args.mask_manifest, out)
        if args.val_manifest
        else []
    )

    def save(epoch, epoch_completed=False, name="last.pt"):
        state = {
            "model": model.state_dict(),
            "ema_model": ema.state_dict() if ema else None,
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "step": step,
            "epoch": epoch,
            "epoch_completed": epoch_completed,
            "best_score": best,
            "validation_metric_version": 2,
            "metrics": last_metrics,
            "model_config": mc,
            "training_config": cfg,
            "data_hashes": identities,
            "anatomy_version": ANATOMY_VERSION
            if mc.get("anatomy_channels", 0)
            else None,
            "method": "conditional_rectified_flow",
            "mask_manifest": args.mask_manifest,
        }
        atomic_save(state, out / name)

    def validate(epoch, completed=False):
        nonlocal best, last_val, last_metrics
        last_metrics = evaluate(ema or model, cases, cfg, device, out, step)
        last_val = step
        improved = last_metrics["selection_score"] < best
        if improved:
            best = last_metrics["selection_score"]
        with (out / "validation_history.jsonl").open("a") as f:
            f.write(json.dumps(last_metrics) + "\n")
        print("validation " + json.dumps(last_metrics), flush=True)
        if improved:
            save(epoch, completed, "best.pt")
        save(epoch, completed)

    if args.preflight and device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    if (
        cases
        and step == 0
        and not args.preflight
        and cfg.get("validate_at_start", True)
    ):
        validate(1)
    for epoch in range(start_epoch, cfg["epochs"]):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running = 0.0
        group_count = 0
        bar = tqdm(loader, desc=f"epoch {epoch + 1}/{cfg['epochs']}")
        for batch_index, batch in enumerate(bar):
            context = batch["image"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            anatomy = batch.get("anatomy")
            anatomy = (
                anatomy.to(device, non_blocking=True) if anatomy is not None else None
            )
            noise = torch.randn_like(target)
            t = torch.rand(target.shape[0], device=device)
            tv = t.view(-1, 1, 1, 1, 1)
            sample = torch.where(mask > 0, (1 - tv) * noise + tv * target, context)
            # Short final groups get their actual divisor, not a smaller effective LR.
            group_start = (batch_index // accumulation) * accumulation
            divisor = min(accumulation, len(loader) - group_start)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda" and cfg["amp"],
            ):
                velocity = model(
                    sample, context, mask, t * cfg["time_embedding_scale"], anatomy
                )
                loss, terms = training_loss(
                    velocity, target - noise, sample, target, mask, t, cfg
                )
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite training loss")
            scaler.scale(loss / divisor).backward()
            group_count += 1
            running += float(loss.detach())
            if group_count < divisor:
                continue
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                cfg["grad_clip"],
                error_if_nonfinite=not scaler.is_enabled(),
            )
            warmup = cfg.get("warmup_steps", 0)
            fraction = (
                min(1.0, (step + 1) / warmup) if warmup and step < warmup else 1.0
            )
            if cfg.get("cosine_decay", False) and step >= warmup:
                progress = min(1.0, (step - warmup) / max(1, max_steps - warmup))
                floor = cfg.get("minimum_lr_ratio", 0.1)
                fraction = floor + (1 - floor) * 0.5 * (
                    1 + math.cos(math.pi * progress)
                )
            for group in optimizer.param_groups:
                group["lr"] = cfg["learning_rate"] * fraction
            old_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            updated = scaler.get_scale() >= old_scale
            optimizer.zero_grad(set_to_none=True)
            group_count = 0
            if not updated:
                print("AMP skipped a nonfinite optimizer update", flush=True)
                continue
            step += 1
            if ema:
                decay = min(cfg["ema_decay"], (1 + step) / (10 + step))
                with torch.no_grad():
                    for average, parameter in zip(ema.parameters(), model.parameters()):
                        average.lerp_(parameter, 1 - decay)
                    for average, value in zip(ema.buffers(), model.buffers()):
                        average.copy_(value)
            bar.set_postfix(step=step, loss=f"{float(loss.detach()):.4f}")
            if step % cfg.get("log_every_steps", 25) == 0 or args.preflight:
                with (out / "training_history.jsonl").open("a") as f:
                    f.write(
                        json.dumps(
                            {
                                "step": step,
                                "epoch": epoch + 1,
                                "lr": optimizer.param_groups[0]["lr"],
                                "loss": float(loss.detach()),
                                "grad_norm": float(grad_norm),
                                **terms,
                            }
                        )
                        + "\n"
                    )
            if args.preflight:
                if cases:
                    short = dict(cfg, validation_steps=4, validation_seeds=1)
                    metrics = evaluate(
                        ema or model, cases[:1], short, device, out, step
                    )
                    print("preflight_validation " + json.dumps(metrics), flush=True)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                    print(
                        f"preflight_peak_gib={torch.cuda.max_memory_allocated() / 2**30:.3f}",
                        flush=True,
                    )
                print(
                    f"preflight_passed device={device} patch={cfg['patch_size']} parameters={sum(p.numel() for p in model.parameters())}",
                    flush=True,
                )
                return
            if (
                cfg.get("checkpoint_every_steps", 500) > 0
                and step % cfg.get("checkpoint_every_steps", 500) == 0
            ):
                save(epoch + 1)
            if cases and step % cfg.get("validation_every_steps", 2000) == 0:
                validate(epoch + 1)
            if step >= max_steps:
                break
        completed = batch_index == len(loader) - 1
        if cases and last_val != step:
            validate(epoch + 1, completed)
        save(epoch + 1, completed)
        if cfg.get("keep_epoch_checkpoints", True):
            save(epoch + 1, completed, f"epoch_{epoch + 1:03d}.pt")
        print(
            f"epoch={epoch + 1} step={step} mean_total_loss={running / max(1, batch_index + 1):.6f}",
            flush=True,
        )
        if step >= max_steps:
            break


if __name__ == "__main__":
    main()
