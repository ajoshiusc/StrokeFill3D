"""Train/resume 96-cube conditional RF with domain-specific generation validation."""

import argparse
import copy
import json
import math
import random
import socket
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader

from strokefill3d.normal_prior.core import coordinate_maps, file_hash, make_prior
from strokefill3d.normal_prior.data import CounterSampler, cortical_mask
from strokefill3d.normal_prior.io import (
    RunLock,
    atomic_json,
    atomic_save,
    cpu_state,
    restore_rng,
    rng_state,
)
from strokefill3d.validation import reconstruction_metrics

from .core import sample, velocity
from .data import (
    MixedPatches,
    cohort_summary,
    observed_validation_target,
    validate_splits,
)


def fixed_validation(ds, count):
    cases = []
    for domain in ds.domains:
        seen = set()
        sub = ds.datasets[domain]
        for i in range(count):
            for attempt in range(100):
                case = sub[1000000 + i * 1009 + attempt * 101]
                if case["source"] in seen:
                    continue
                try:
                    mask = cortical_mask(
                        case["target"],
                        case["origin"],
                        ds.donors,
                        np.random.default_rng(909 + i + attempt),
                        real_only=True,
                    )
                except ValueError:
                    continue
                case.update(mask=mask, domain=domain)
                case["target"] = observed_validation_target(case, sub.rows)
                cases.append(case)
                seen.add(case["source"])
                break
            else:
                raise RuntimeError(f"Cannot build validation in {domain}")
    return cases


@torch.no_grad()
def evaluate(model, cases, device, out, step):
    model.eval()
    records = []
    panels = []
    for i, c in enumerate(cases):
        target = c["target"][None].to(device)
        mask = c["mask"][None].to(device)
        image = torch.where(mask > 0, 0, target)
        scores = []
        for seed in [909, 910]:
            pred = sample(model, image, mask, c["origin"][None].to(device), seed)
            scores.append(reconstruction_metrics(pred, target, mask))
            if seed == 909:
                # Display the coronal plane with the largest hidden region.
                y = int(mask[0, 0].sum((0, 2)).argmax())
                row = []
                for v in [target, image, pred]:
                    a = v[0, 0, :, y, :].cpu().numpy()
                    if v is image:
                        a = np.where(mask[0, 0, :, y, :].cpu().numpy() > 0, -1, a)
                    row.append(
                        Image.fromarray(
                            np.uint8(np.clip((np.flipud(a.T) + 1) * 127.5, 0, 255))
                        ).resize((192, 192))
                    )
                panels.append((c["domain"], row))
        record = {"domain": c["domain"], "source": c["source"], "case": i}
        record.update(
            {
                k: float(np.mean([s[k] for s in scores]))
                for k in scores[0]
                if all(s[k] is not None for s in scores)
            }
        )
        records.append(record)
    domains = {}
    for domain in sorted({r["domain"] for r in records}):
        rows = [r for r in records if r["domain"] == domain]
        domains[domain] = {
            k: float(np.mean([r[k] for r in rows]))
            for k in ["masked_mae", "masked_ssim", "gradient_mae", "outside_max_error"]
        }
    # Equal domain weights; a large ATLAS cohort cannot hide external regression.
    score = float(
        np.mean(
            [
                r["masked_mae"]
                + 0.25 * (1 - r["masked_ssim"])
                + 0.25 * r["gradient_mae"]
                for r in domains.values()
            ]
        )
    )
    result = {"step": step, "score": score, "domains": domains, "cases": records}
    atomic_json(result, out / "metrics.json")
    canvas = Image.new("RGB", (616, 40 + len(panels) * 218), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (8, 8),
        "Complete target             Hidden input                 Generated RF",
        fill="black",
    )
    for i, (domain, row) in enumerate(panels):
        y = 40 + i * 218
        draw.text((8, y), domain, fill="black")
        for j, img in enumerate(row):
            canvas.paste(img, (8 + j * 202, y + 20))
    canvas.save(out / "comparison.png")
    print(
        "VALIDATION",
        json.dumps({"step": step, "score": score, "domains": domains}),
        flush=True,
    )
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--preflight", action="store_true")
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-steps", type=int)
    p.add_argument(
        "--patch-size",
        type=int,
        help="Functional smoke test only; fresh directory required",
    )
    p.add_argument(
        "--evaluate-test",
        action="store_true",
        help="Evaluate best checkpoint on untouched test manifest",
    )
    args = p.parse_args()
    with open(args.config) as stream:
        cfg = yaml.safe_load(stream)
    if args.patch_size:
        cfg["patch_size"] = [args.patch_size] * 3
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; run outside the sandbox")
    torch.set_num_threads(cfg["cpu_threads"])
    torch.manual_seed(cfg["seed"])
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    train, val, test = [args.data / (s + ".csv") for s in ["train", "val", "test"]]
    validate_splits(train, val, test)
    counts = cohort_summary(args.data, require_ready=not args.preflight)
    print("COHORT", json.dumps(counts, sort_keys=True), flush=True)
    identity = {
        "version": "mixed_conditional_image_rf_3d_v1",
        "cohort_counts": counts,
        "config": cfg,
        "functional_only": args.preflight or bool(args.patch_size),
        "data_hashes": {
            s: file_hash(args.data / (s + ".csv")) for s in ["train", "val", "test"]
        },
    }
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    with RunLock(out):
        meta = out / "run.json"
        if meta.exists():
            old = json.loads(meta.read_text())
            if not (args.resume or args.evaluate_test) or any(
                old[k] != v for k, v in identity.items()
            ):
                raise ValueError(
                    "Run identity differs or --resume missing; use a fresh run directory"
                )
        else:
            if args.resume or args.evaluate_test:
                raise ValueError("No run to resume/evaluate")
            if any(
                p.name not in [".normal_prior.lock", "train.log"] for p in out.iterdir()
            ):
                raise ValueError("Output is not empty")
            atomic_json(dict(identity, host=socket.gethostname()), meta)
        ds = MixedPatches(train, cfg["patch_size"], cfg["seed"], cfg["volume_cache"])
        vd = MixedPatches(
            test if args.evaluate_test else val, cfg["patch_size"], cfg["seed"] + 1, 1
        )
        cases_path = out / (
            "test_cases.pt" if args.evaluate_test else "validation_cases.pt"
        )
        if cases_path.exists():
            cases = torch.load(cases_path, weights_only=True)
        else:
            cases = fixed_validation(
                vd, 1 if args.preflight else cfg["validation_per_domain"]
            )
            atomic_save(cases, cases_path)
        model = make_prior(cfg["model"]).to(device)
        if args.evaluate_test:
            saved = torch.load(out / "best.pt", map_location="cpu", weights_only=True)
            model.load_state_dict(saved["model"])
            evaluate(model, cases, device, out / "test", saved["step"])
            return
        ema = copy.deepcopy(model).eval().requires_grad_(False)
        opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=1e-4)
        scaler = torch.amp.GradScaler(
            device.type, enabled=device.type == "cuda", init_scale=1024
        )
        start = 0
        best = float("inf")
        updates = 0
        if args.resume:
            s = torch.load(out / "latest.pt", map_location="cpu", weights_only=True)
            if s["identity"] != identity:
                raise ValueError("Checkpoint identity differs")
            model.load_state_dict(s["online"])
            ema.load_state_dict(s["model"])
            opt.load_state_dict(s["optimizer"])
            scaler.load_state_dict(s["scaler"])
            start = s["step"]
            best = s["best_score"]
            updates = s["updates"]
            restore_rng(s)
        total = min(cfg["steps"], args.max_steps or cfg["steps"])
        if args.preflight:
            total = start + 1
        accumulation = 1 if args.preflight else cfg["accumulation"]
        consumed = start * cfg["batch_size"] * accumulation
        loader = DataLoader(
            ds,
            batch_size=cfg["batch_size"],
            sampler=CounterSampler(
                consumed, (total - start) * cfg["batch_size"] * accumulation
            ),
            num_workers=0 if args.preflight else cfg["workers"],
            pin_memory=device.type == "cuda",
            generator=torch.Generator().manual_seed(cfg["seed"]),
        )
        it = iter(loader)
        tic = time.time()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        for step in range(start + 1, total + 1):
            model.train()
            opt.zero_grad(set_to_none=True)
            loss_sum = 0.0
            for group in opt.param_groups:
                group["lr"] = (
                    cfg["lr"]
                    * min(1, step / 500)
                    * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * step / cfg["steps"])))
                )
            for _ in range(accumulation):
                batch = next(it)
                target = batch["target"].to(device)
                mask = batch["mask"].to(device)
                # The 2D winner used mild gamma/gain; no histogram/bias augmentation.
                gamma = 0.9 + 0.2 * torch.rand((len(target), 1, 1, 1, 1), device=device)
                gain = 0.95 + 0.1 * torch.rand_like(gamma)
                target = torch.where(
                    target > -0.95,
                    (((target + 1) / 2).clamp_min(1e-5).pow(gamma) * gain).clamp(0, 1)
                    * 2
                    - 1,
                    target,
                )
                image = torch.where(mask > 0, 0, target)
                noise = torch.randn_like(target)
                t = torch.rand(len(target), device=device) * 0.95
                state = torch.where(
                    mask > 0,
                    (1 - t[:, None, None, None, None]) * noise
                    + t[:, None, None, None, None] * target,
                    image,
                )
                coords = coordinate_maps(
                    batch["origin"].to(device), target.shape[2:], 1
                )
                v = velocity(model, state, image, mask, coords, t)
                loss = (
                    ((v - target + noise).square() * mask).flatten(1).sum(1)
                    / mask.flatten(1).sum(1).clamp_min(1)
                ).mean() / accumulation
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite loss")
                scaler.scale(loss).backward()
                loss_sum += float(loss.detach())
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            before = scaler.get_scale()
            scaler.step(opt)
            scaler.update()
            if scaler.get_scale() >= before:
                updates += 1
                with torch.no_grad():
                    torch._foreach_lerp_(
                        list(ema.parameters()),
                        list(model.parameters()),
                        1 - min(0.999, (updates + 1) / (updates + 10)),
                    )
            if step % 20 == 0 or args.preflight:
                print(
                    f"step={step} loss={loss_sum:.5f} elapsed_min={(time.time() - tic) / 60:.1f}",
                    flush=True,
                )
            metrics = None
            if step % cfg["validate_every"] == 0 or step == total:
                metrics = evaluate(
                    ema, cases, device, out / f"validation/step_{step:08d}", step
                )
                with (out / "validation_history.jsonl").open("a") as f:
                    f.write(json.dumps(metrics) + "\n")
            improved = metrics is not None and metrics["score"] < best
            if improved:
                best = metrics["score"]
            if step % cfg["save_every"] == 0 or metrics is not None:
                saved = dict(
                    kind="mixed_conditional_image_rf_3d_v1",
                    step=step,
                    updates=updates,
                    identity=identity,
                    model_config=cfg["model"],
                    patch_size=cfg["patch_size"],
                    model=cpu_state(ema),
                    online=cpu_state(model),
                    optimizer=opt.state_dict(),
                    scaler=scaler.state_dict(),
                    best_score=best,
                    **rng_state(),
                )
                atomic_save(saved, out / "latest.pt")
                if improved:
                    atomic_save(saved, out / "best.pt")
                if step % 10000 == 0:
                    atomic_save(saved, out / f"step_{step:08d}.pt")
        atomic_json(
            {
                "step": total,
                "seconds": time.time() - tic,
                "peak_gpu_gib": torch.cuda.max_memory_allocated() / 2**30
                if device.type == "cuda"
                else None,
                "functional_only": args.preflight or bool(args.patch_size),
                "successful_optimizer_updates": updates,
            },
            out / "status.json",
        )
        print("Finished", str(out), flush=True)


if __name__ == "__main__":
    main()
