"""Two-stage 96-cube normal-code RF training; all data/model identities are locked."""

import argparse
import copy
import io
import json
import math
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from monai.metrics.regression import compute_ssim_and_cs
from torch.utils.data import DataLoader, Dataset, Sampler

from strokefill3d.validation import check_split

from .core import (
    VERSION,
    Codec,
    amp,
    coordinate_maps,
    file_hash,
    make_autoencoder,
    make_prior,
    prior_config,
    velocity,
)
from .data import CounterSampler, NormalPatches, fixed_cases
from .evaluation import evaluate_codec, evaluate_prior, quality_passed
from .io import (
    RunLock,
    atomic_json,
    atomic_save,
    cpu_state,
    initialize_run,
    restore_rng,
    rng_state,
)


def log(message):
    print(message, flush=True)


def loader(dataset, start, count, cfg, batch=1, sampler=None):
    return DataLoader(
        dataset,
        batch_size=batch,
        sampler=sampler or CounterSampler(start, count),
        num_workers=cfg.get("workers", 0),
        pin_memory=True,
        persistent_workers=cfg.get("workers", 0) > 0,
    )


def optimizer_for(model, lr, device):
    return torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=1e-4,
        fused=device.type == "cuda",
    )


def restore_optimizer(optimizer, state, device):
    optimizer.load_state_dict(state)
    for group in optimizer.param_groups:
        group["fused"] = device.type == "cuda"
        group["foreach"] = False
    for value in optimizer.state.values():
        if torch.is_tensor(value.get("step")):
            value["step"] = value["step"].to(device)


def update_ema(ema, model, step, decay):
    with torch.no_grad():
        torch._foreach_lerp_(
            list(ema.parameters()),
            list(model.parameters()),
            1 - min(decay, (step + 1) / (step + 10)),
        )


def reconstruction_loss(pred, target):
    weight = 0.25 + 0.75 * (target > -0.95)
    l1 = ((pred - target).abs() * weight).sum() / weight.sum()
    ss, _ = compute_ssim_and_cs(
        F.pad(pred, [3, 3] * 3, mode="reflect"),
        F.pad(target, [3, 3] * 3, mode="reflect"),
        3,
        (7,) * 3,
        (1.5,) * 3,
        data_range=2.0,
    )
    similarity = ((1 - ss) * weight).sum() / weight.sum()
    gradient = 0
    for dim in (2, 3, 4):
        gradient = (
            gradient + (pred.diff(dim=dim) - target.diff(dim=dim)).abs().mean() / 3
        )
    return l1 + 0.3 * similarity + 0.5 * gradient


def ae_step(model, target, step, cfg):
    with amp(target.device):
        mu, sigma = model.encode(target)
        std = mu.detach().float().std(dim=(0, 2, 3, 4), keepdim=True).clamp_min(0.01)
        level = min(1, step / 2000) * cfg.get("latent_noise", 0.1)
        amplitude = torch.rand(len(mu), 1, 1, 1, 1, device=mu.device) * level
        amplitude = torch.where(torch.rand_like(amplitude) < 0.25, 0, amplitude)
        pred = model.decode(mu + torch.randn_like(mu) * std * amplitude).float()
        kl = (
            0.5
            * (
                mu.float().square()
                + sigma.float().square()
                - 2 * sigma.float().clamp_min(1e-8).log()
                - 1
            ).mean()
        )
    return reconstruction_loss(pred, target) + 1e-6 * kl


def train_ae(cfg, dataset, cases, out, device, resume=False, max_steps=None):
    out = Path(out)
    stage = out / "ae"
    stage.mkdir(exist_ok=True)
    settings = cfg["autoencoder"]
    total = max_steps or settings["steps"]
    if (out / "codec.pt").exists():
        if max_steps:
            raise ValueError(
                "Codec is already frozen. Extending AE would invalidate RF codes; use a new run"
            )
        return
    model = make_autoencoder(settings["model"]).to(device)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    optimizer = optimizer_for(model, settings["lr"], device)
    scaler = torch.amp.GradScaler(
        device.type, enabled=device.type == "cuda", init_scale=1024
    )
    start = 0
    best = float("inf")
    checkpoint = stage / "last.pt"
    if resume and checkpoint.exists():
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if (
            saved.get("kind") != VERSION + "_ae"
            or saved["ae_config"] != settings["model"]
        ):
            raise ValueError("Incompatible AE checkpoint")
        model.load_state_dict(saved["online"])
        ema.load_state_dict(saved["model"])
        restore_optimizer(optimizer, saved["optimizer"], device)
        scaler.load_state_dict(saved["scaler"])
        start = saved["step"]
        best = saved["best_score"]
        restore_rng(saved)
    if max_steps and total < start:
        raise ValueError("Requested AE limit is below the saved step")
    batches = iter(loader(dataset, start, max(0, total - start), cfg["data"]))
    begin = time.time()
    for step in range(start + 1, total + 1):
        target = next(batches)["target"].to(device)
        optimizer.zero_grad(set_to_none=True)
        for g in optimizer.param_groups:
            g["lr"] = (
                settings["lr"]
                * min(1, step / 200)
                * (0.15 + 0.85 * 0.5 * (1 + math.cos(math.pi * min(step / total, 1))))
            )
        loss = ae_step(model, target, step, settings)
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite AE loss")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1, foreach=True)
        old = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        if scaler.get_scale() >= old:
            update_ema(ema, model, step, 0.999)
        if step % 50 == 0:
            log(
                f"AE step={step}/{total} loss={float(loss):.5f} elapsed={time.time() - begin:.0f}s"
            )
        validation = step % settings["validate_every"] == 0 or step == total
        result = (
            evaluate_codec(ema, cases, device, stage / f"validation_{step:08d}")
            if validation
            else None
        )
        improved = False
        if result:
            score = result["masked_mae"] + 0.25 * (1 - result["masked_ssim"])
            improved = score < best
            if improved:
                best = score
            log("AE_VALIDATION " + json.dumps(result))
            atomic_json(result, stage / "latest_validation.json")
        if validation or step % settings["save_every"] == 0:
            saved = dict(
                kind=VERSION + "_ae",
                model=cpu_state(ema),
                online=cpu_state(model),
                ae_config=settings["model"],
                patch_size=cfg["patch_size"],
                data_hashes=json.loads((out / "run.json").read_text())["data_hashes"],
                step=step,
                best_score=best,
                optimizer=optimizer.state_dict(),
                scaler=scaler.state_dict(),
                metrics=result,
                **rng_state(),
            )
            atomic_save(saved, checkpoint)
            if improved:
                atomic_save(saved, stage / "best.pt")
    del batches, model, ema, optimizer
    torch.cuda.empty_cache()
    bestck = torch.load(stage / "best.pt", map_location="cpu", weights_only=True)
    if not quality_passed(bestck["metrics"], cfg["codec_gate"]):
        raise RuntimeError(
            "AE reconstruction gate has not passed. RF training was not started. Inspect ae/validation_* and extend AE training with --stage ae --resume --max-steps N"
        )


@torch.inference_mode()
def freeze_codec(cfg, dataset, cases, out, device):
    out = Path(out)
    path = out / "codec.pt"
    best_path = out / "ae/best.pt"
    if path.exists():
        saved = torch.load(path, map_location="cpu", weights_only=True)
        if saved["ae_sha256"] != file_hash(best_path):
            raise ValueError("Frozen codec was built from a different AE checkpoint")
        return Codec(saved).to(device)
    best = torch.load(best_path, map_location="cpu", weights_only=True)
    if not quality_passed(best["metrics"], cfg["codec_gate"]):
        raise ValueError("AE capacity gate has not passed")
    ae = make_autoencoder(best["ae_config"]).to(device).eval().requires_grad_(False)
    ae.load_state_dict(best["model"])
    total = 0
    sum_mu = None
    cross = None
    count = cfg["calibration_samples"]
    batches = iter(loader(dataset, 10_000_000, count, cfg["data"]))
    for _ in range(count):
        target = next(batches)["target"].to(device)
        with amp(device):
            mu, _ = ae.encode(target)
        points = mu.float().cpu().movedim(1, 0).flatten(1).double()
        s = points.sum(1)
        c = points @ points.T
        sum_mu = s if sum_mu is None else sum_mu + s
        cross = c if cross is None else cross + c
        total += points.shape[1]
    mean = sum_mu / total
    cov = (cross - total * mean[:, None] * mean[None, :]) / max(1, total - 1)
    std = cov.diag().clamp_min(1e-8).sqrt()
    correlation = cov / (std[:, None] * std[None, :])
    values, vectors = torch.linalg.eigh(correlation)
    keep = values > values.max() * cfg.get("pca_relative_floor", 1e-4)
    if not keep.any():
        raise ValueError("Degenerate AE code statistics")
    basis = vectors[:, keep]
    scales = values[keep].sqrt()
    whiten = basis / scales[None, :]
    unwhiten = scales[:, None] * basis.T
    factor = 2 ** (len(best["ae_config"]["channels"]) - 1)
    saved = {
        "kind": VERSION + "_codec",
        "ae_config": best["ae_config"],
        "model": best["model"],
        "patch_size": cfg["patch_size"],
        "factor": factor,
        "mean": mean.float()[None, :, None, None, None],
        "std": std.float()[None, :, None, None, None],
        "whiten": whiten.float(),
        "unwhiten": unwhiten.float(),
        "eigenvalues": values.float(),
        "retained_variance": float(values[keep].sum() / values.sum()),
        "ae_sha256": file_hash(best_path),
        "calibration_samples": count,
        "quality": {},
    }
    codec = Codec(saved).to(device)
    result = evaluate_codec(codec, cases, device, out / "codec_validation", codec=True)
    result["passed"] = quality_passed(result, cfg["codec_gate"])
    saved["quality"] = result
    atomic_json(result, out / "codec_quality.json")
    if not result["passed"]:
        raise RuntimeError(
            "The calibrated/truncated codec lost detail. RF/cache stages were not started"
        )
    atomic_save(saved, path)
    log(
        f"CODEC passed: {codec.channels} channels, factor={factor}, latent={[n // factor for n in cfg['patch_size']]}"
    )
    return codec


class CodeCache(Dataset):
    def __init__(self, path, codec_sha, train_sha):
        self.path = Path(path)
        self.meta = json.loads((self.path / "complete.json").read_text())
        self.cache = OrderedDict()
        if (
            self.meta["codec_sha256"] != codec_sha
            or self.meta["train_sha256"] != train_sha
        ):
            raise ValueError("Code cache/codec/training manifest fingerprints differ")
        self.length = sum(s["count"] for s in self.meta["shards"])
        self.offsets = np.cumsum([0] + [s["count"] for s in self.meta["shards"]])

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        shard = int(np.searchsorted(self.offsets, index, side="right") - 1)
        local = index - int(self.offsets[shard])
        if shard not in self.cache:
            meta = self.meta["shards"][shard]
            raw = (self.path / meta["name"]).read_bytes()
            import hashlib

            if hashlib.sha256(raw).hexdigest() != meta["sha256"]:
                raise ValueError("Corrupt code shard")
            data = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
            if (
                data["codec_sha256"] != self.meta["codec_sha256"]
                or not torch.isfinite(data["codes"]).all()
            ):
                raise ValueError("Invalid cached code shard")
            self.cache[shard] = data
            while len(self.cache) > 2:
                self.cache.popitem(last=False)
        self.cache.move_to_end(shard)
        data = self.cache[shard]
        return {"code": data["codes"][local].float(), "origin": data["origins"][local]}


class ShardSampler(Sampler):
    def __init__(self, dataset, start, count, seed):
        self.data = dataset
        self.start = start
        self.count = count
        self.seed = seed

    def __len__(self):
        return self.count

    def __iter__(self):
        epoch, offset = divmod(self.start, len(self.data))
        remaining = self.count
        while remaining:
            rng = np.random.default_rng(self.seed + epoch)
            order = []
            for shard in rng.permutation(len(self.data.meta["shards"])):
                begin, end = self.data.offsets[shard : shard + 2]
                order.extend((begin + rng.permutation(end - begin)).tolist())
            for value in order[offset:]:
                yield int(value)
                remaining -= 1
                if not remaining:
                    return
            epoch += 1
            offset = 0


@torch.inference_mode()
def cache_codes(cfg, dataset, out, device, train_sha):
    out = Path(out)
    codec_path = out / "codec.pt"
    codec = Codec(torch.load(codec_path, map_location="cpu", weights_only=True)).to(
        device
    )
    if not codec.quality.get("passed"):
        raise ValueError("Unqualified codec")
    path = out / "codes"
    path.mkdir(exist_ok=True)
    sha = file_hash(codec_path)
    identity = {
        "codec_sha256": sha,
        "train_sha256": train_sha,
        "count": cfg["cache"]["samples"],
        "shard_size": cfg["cache"]["shard_size"],
        "seed": cfg["seed"],
    }
    meta_path = path / "identity.json"
    if meta_path.exists() and json.loads(meta_path.read_text()) != identity:
        raise ValueError("Code cache identity changed")
    atomic_json(identity, meta_path)
    if (path / "complete.json").exists():
        return CodeCache(path, sha, train_sha)
    shards = []
    size = identity["shard_size"]
    stream = None
    next_index = None
    for begin in range(0, identity["count"], size):
        count = min(size, identity["count"] - begin)
        name = f"shard_{begin // size:05d}.pt"
        file = path / name
        if not file.exists():
            if next_index != begin:
                stream = iter(
                    loader(
                        dataset,
                        20_000_000 + begin,
                        identity["count"] - begin,
                        cfg["data"],
                    )
                )
                next_index = begin
            codes = []
            origins = []
            sources = []
            for _ in range(count):
                batch = next(stream)
                next_index += 1
                codes.append(codec.encode(batch["target"].to(device)).half().cpu())
                origins.append(batch["origin"])
                sources.extend(batch["source"])
            atomic_save(
                {
                    "codes": torch.cat(codes),
                    "origins": torch.cat(origins),
                    "sources": sources,
                    "codec_sha256": sha,
                },
                file,
            )
        saved = torch.load(file, map_location="cpu", weights_only=True)
        if saved["codec_sha256"] != sha or len(saved["codes"]) != count:
            raise ValueError("Wrong existing code shard")
        shards.append({"name": name, "count": count, "sha256": file_hash(file)})
        log(f"CACHE {begin + count}/{identity['count']}")
    atomic_json(dict(**identity, shards=shards), path / "complete.json")
    return CodeCache(path, sha, train_sha)


def train_prior(cfg, cache, cases, out, device, resume=False, max_steps=None):
    out = Path(out)
    settings = cfg["prior"]
    stage = out / "prior"
    stage.mkdir(exist_ok=True)
    total = max_steps or settings["steps"]
    codec_sha = file_hash(out / "codec.pt")
    codec = Codec(
        torch.load(out / "codec.pt", map_location="cpu", weights_only=True)
    ).to(device)
    if not codec.quality.get("passed"):
        raise ValueError("Codec quality gate has not passed")
    mc = prior_config(settings["model"], codec.channels)
    model = make_prior(mc).to(device)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    optimizer = optimizer_for(model, settings["lr"], device)
    scaler = torch.amp.GradScaler(
        device.type, enabled=device.type == "cuda", init_scale=1024
    )
    start = 0
    best = float("inf")
    accum = settings["accumulation"]
    batch_size = settings["batch_size"]
    micro = accum * batch_size
    if resume and (stage / "last.pt").exists():
        saved = torch.load(stage / "last.pt", map_location="cpu", weights_only=True)
        if (
            saved.get("kind") != VERSION + "_prior"
            or saved["codec_sha256"] != codec_sha
            or saved["model_config"] != mc
        ):
            raise ValueError("RF resume model/codec differs")
        model.load_state_dict(saved["online"])
        ema.load_state_dict(saved["model"])
        restore_optimizer(optimizer, saved["optimizer"], device)
        scaler.load_state_dict(saved["scaler"])
        start = saved["step"]
        best = saved["best_score"]
        restore_rng(saved)
    if max_steps and total < start:
        raise ValueError("Requested RF limit is below the saved step")
    if start >= total and (stage / "final_full_validation/metrics.json").exists():
        log("RF training and final validation already complete")
        return
    sample = ShardSampler(
        cache, start * micro, max(0, total - start) * micro, cfg["seed"]
    )
    batches = iter(
        loader(
            cache,
            0,
            len(sample),
            {"workers": cfg["cache"]["workers"]},
            batch_size,
            sample,
        )
    )
    begin = time.time()
    for step in range(start + 1, total + 1):
        optimizer.zero_grad(set_to_none=True)
        model.train()
        loss_value = 0
        for group in optimizer.param_groups:
            group["lr"] = (
                settings["lr"]
                * min(1, step / 500)
                * (0.15 + 0.85 * 0.5 * (1 + math.cos(math.pi * min(step / total, 1))))
            )
        for _ in range(accum):
            batch = next(batches)
            target = batch["code"].to(device)
            coords = coordinate_maps(
                batch["origin"].to(device), target.shape[2:], codec.factor
            )
            noise = torch.randn_like(target)
            t = torch.rand(len(target), device=device)
            tv = t[:, None, None, None, None]
            state = (1 - tv) * noise + tv * target
            v = velocity(model, state, coords, t)
            loss = (v - target + noise).square().mean() / accum
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite RF loss")
            scaler.scale(loss).backward()
            loss_value += float(loss.detach())
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1, foreach=True)
        old = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        if scaler.get_scale() >= old:
            update_ema(ema, model, step, 0.9999)
        if step % 50 == 0:
            log(
                f"RF step={step}/{total} velocity_mse={loss_value:.5f} elapsed={time.time() - begin:.0f}s"
            )
        validation = step % settings["validate_every"] == 0 or step == total
        result = None
        improved = False
        if validation:
            result = evaluate_prior(
                ema,
                codec,
                cases,
                device,
                stage / f"validation_{step:08d}",
                cfg["validation"]["quick"],
                step,
            )
            score = (
                result["masked_mae"]
                + 0.25 * (1 - result["masked_ssim"])
                + 0.25 * result["gradient_mae"]
            )
            improved = score < best
            if improved:
                best = score
            log(
                "RF_VALIDATION "
                + json.dumps(
                    {
                        k: result[k]
                        for k in ["step", "masked_mae", "masked_ssim", "gradient_mae"]
                    }
                )
            )
        if validation or step % settings["save_every"] == 0:
            saved = dict(
                kind=VERSION + "_prior",
                model=cpu_state(ema),
                online=cpu_state(model),
                model_config=mc,
                codec_sha256=codec_sha,
                step=step,
                best_score=best,
                optimizer=optimizer.state_dict(),
                scaler=scaler.state_dict(),
                metrics=result,
                patch_size=cfg["patch_size"],
                **rng_state(),
            )
            atomic_save(saved, stage / "last.pt")
            if improved:
                atomic_save(saved, stage / "best.pt")
    del batches, model, ema, optimizer
    torch.cuda.empty_cache()
    from .core import load_bundle

    model, codec, saved = load_bundle(out, device=device)
    evaluate_prior(
        model,
        codec,
        cases,
        device,
        stage / "final_full_validation",
        cfg["validation"]["full"],
        saved["step"],
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--val-manifest", required=True)
    p.add_argument("--test-manifest")
    p.add_argument("--mask-manifest")
    p.add_argument("--output", required=True)
    p.add_argument(
        "--stage",
        choices=["all", "ae", "freeze", "cache", "prior", "evaluate", "preflight"],
        default="all",
    )
    p.add_argument("--resume", action="store_true")
    p.add_argument("--max-steps", type=int)
    p.add_argument("--device", default="cuda")
    p.add_argument("--reuse-ae")
    args = p.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    device = torch.device(args.device)
    torch.set_num_threads(cfg.get("cpu_threads", 4))
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable")
    if len(cfg["patch_size"]) != 3 or any(n % 16 for n in cfg["patch_size"]):
        raise ValueError("Patch must be 3D and divisible by 16")
    if args.max_steps is not None and args.max_steps < 1:
        raise ValueError("Step limit must be positive")
    if args.max_steps and args.stage not in ["ae", "prior"]:
        raise ValueError("--max-steps applies to --stage ae or prior only")
    if args.reuse_ae and (args.resume or args.stage not in ["all", "freeze"]):
        raise ValueError("--reuse-ae is for a new all/freeze run")
    check_split(
        args.manifest, args.val_manifest, args.mask_manifest, args.test_manifest
    )
    out = Path(args.output)
    with RunLock(out):
        if args.stage == "preflight":
            from .preflight import run

            run(cfg, args, out / "preflight", device)
            return
        paths = {
            "train": args.manifest,
            "validation": args.val_manifest,
            "test": args.test_manifest,
            "masks": args.mask_manifest,
        }
        identity = initialize_run(out, cfg, paths, args.resume)
        if args.reuse_ae:
            saved = torch.load(args.reuse_ae, map_location="cpu", weights_only=True)
            if (
                saved.get("kind") != VERSION + "_ae"
                or saved["ae_config"] != cfg["autoencoder"]["model"]
                or saved["patch_size"] != cfg["patch_size"]
                or saved.get("data_hashes") != identity["data_hashes"]
            ):
                raise ValueError(
                    "Reused AE architecture, patch or data manifests differ"
                )
            atomic_save(saved, out / "ae/best.pt")
            atomic_save(saved, out / "ae/last.pt")
        dataset = NormalPatches(
            args.manifest,
            cfg["patch_size"],
            cfg["seed"],
            cfg["data"]["exclusion_margin"],
            cfg["data"]["volume_cache"],
        )
        cases_path = out / "validation_cases.pt"
        if cases_path.exists():
            cases = torch.load(cases_path, map_location="cpu", weights_only=True)
        else:
            val = NormalPatches(
                args.val_manifest,
                cfg["patch_size"],
                cfg["seed"] + 1,
                cfg["data"]["exclusion_margin"],
                1,
            )
            cases = fixed_cases(
                val, cfg["validation"]["count"], args.mask_manifest, cfg["seed"] + 2
            )
            atomic_save(cases, cases_path)
            atomic_json(
                [
                    {
                        k: v
                        for k, v in c.items()
                        if k in ["source", "site", "foreground_fraction"]
                    }
                    for c in cases
                ],
                out / "validation_sources.json",
            )
        if args.stage in ["all", "ae"] and not args.reuse_ae:
            train_ae(
                cfg,
                dataset,
                cases,
                out,
                device,
                args.resume,
                args.max_steps if args.stage == "ae" else None,
            )
        if args.stage in ["all", "freeze"]:
            freeze_codec(cfg, dataset, cases, out, device)
        if args.stage in ["all", "cache"]:
            cache_codes(cfg, dataset, out, device, identity["data_hashes"]["train"])
        if args.stage in ["all", "prior"]:
            cache = CodeCache(
                out / "codes",
                file_hash(out / "codec.pt"),
                identity["data_hashes"]["train"],
            )
            train_prior(
                cfg,
                cache,
                cases,
                out,
                device,
                args.resume,
                args.max_steps if args.stage == "prior" else None,
            )
        if args.stage == "evaluate":
            from .core import load_bundle

            model, codec, saved = load_bundle(out, device=device)
            evaluate_prior(
                model,
                codec,
                cases,
                device,
                out / "evaluation_full",
                cfg["validation"]["full"],
                saved["step"],
            )


if __name__ == "__main__":
    main()
