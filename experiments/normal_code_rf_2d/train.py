"""Train a full normal-MRI latent RF prior, without synthetic-mask conditioning."""

from pathlib import Path
import argparse, copy, json, math, os, random, time
import numpy as np
import torch
import torch.nn.functional as F
from monai.networks.nets import AutoencoderKL
from monai.metrics.regression import compute_ssim_and_cs
from core import (
    FrozenCodec,
    MODEL_CONFIG,
    coordinate_maps,
    make_model,
    healthy_sample,
    inpaint,
    predict_velocity,
)

parser = argparse.ArgumentParser()
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--steps", type=int, default=24000)
parser.add_argument("--batch", type=int, default=48)
parser.add_argument("--bank-size", type=int, default=32768)
parser.add_argument("--validate-every", type=int, default=2000)
parser.add_argument("--resume", type=Path)
parser.add_argument("--plane-bank", type=Path)
parser.add_argument("--autoencoder-checkpoint", type=Path)
parser.add_argument("--frozen-codec", type=Path)
parser.add_argument("--validation-pairs", type=Path, required=True)
parser.add_argument("--compile", action="store_true")
parser.add_argument("--bank-cache", type=Path)
args = parser.parse_args()
OUT = args.output
OUT.mkdir(parents=True, exist_ok=True)
START = time.time()
torch.set_num_threads(3)
torch.manual_seed(90231)
random.seed(90231)
np.random.seed(90231)
torch.backends.cudnn.benchmark = True


def log(message):
    print(message, flush=True)
    with (OUT / "progress.log").open("a") as f:
        f.write(message + "\n")


def cpu_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def atomic_save(value, path):
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


cache = args.bank_cache or OUT / "healthy_code_bank.pt"
if args.frozen_codec and not (OUT / "codec.pt").exists():
    import shutil

    shutil.copy2(args.frozen_codec, OUT / "codec.pt")
if not cache.exists():
    if not args.plane_bank or not args.autoencoder_checkpoint:
        parser.error(
            "Creating codes requires --plane-bank and --autoencoder-checkpoint"
        )
    bank = torch.load(args.plane_bank, weights_only=True)
    planes = bank["planes"].float().cuda()
    exclusions = bank["exclusions"].cuda()
    groups = {}
    for item in bank["candidates"]["train"]:
        groups.setdefault(
            int(bank["subject_index"][item[0]])
            if "subject_index" in bank
            else item[0] // 12,
            [],
        ).append(item)
    subjects = sorted(groups)
    if not subjects:
        raise ValueError("No valid training crops")
    # Equal subject weight; every crop is checked against its source lesion margin.
    selected = [
        random.choice(groups[subjects[i % len(subjects)]])
        for i in range(args.bank_size)
    ]
    random.shuffle(selected)
    saved = torch.load(args.autoencoder_checkpoint, weights_only=True)
    ae = AutoencoderKL(**saved["config"]).cuda().eval().requires_grad_(False)
    ae.load_state_dict(saved["model"])
    mean = saved["mean"].cuda()
    std = saved["std"].cuda()
    axis = torch.arange(64, device="cuda")
    xx, zz = torch.meshgrid(axis, axis, indexing="ij")
    codes, starts = [], []
    for begin in range(0, len(selected), 96):
        items = selected[begin : begin + 96]
        ids, x, z = [torch.tensor(v, device="cuda") for v in zip(*items)]
        bi = ids[:, None, None]
        image = planes[bi, 0, x[:, None, None] + xx, z[:, None, None] + zz][:, None]
        ex = exclusions[bi, 0, x[:, None, None] + xx, z[:, None, None] + zz]
        assert not ex.any()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            raw = (ae.encode(image)[0].float() - mean) / std
        codes.append(raw.cpu())
        starts.append(torch.stack([x, bank["world_y"].cuda()[ids], z], dim=1).cpu())
        if begin % 3072 == 0:
            log(f"ENCODE {begin}/{len(selected)} elapsed={time.time() - START:.0f}s")
    raw = torch.cat(codes).float()
    points = raw.permute(1, 0, 2, 3).flatten(1).double()
    center = points.mean(1).float()[None, :, None, None]
    cov = torch.cov(points).float()
    values, vectors = torch.linalg.eigh(cov)
    basis = vectors[:, 1:]
    scales = values[1:].sqrt().clamp_min(0.02)
    whiten = basis / scales[None, :]
    unwhiten = scales[:, None] * basis.T
    saved.update(center=center, whiten=whiten, unwhiten=unwhiten, eigenvalues=values)
    torch.save(saved, OUT / "codec.pt")
    latent = torch.einsum("bchw,cd->bdhw", raw - center, whiten).half()
    data = dict(
        codes=latent,
        starts=torch.cat(starts).float(),
        subjects=len(subjects),
        eigenvalues=values,
    )
    torch.save(data, cache)
    del bank, planes, exclusions, ae, raw, points, codes
    torch.cuda.empty_cache()
else:
    data = torch.load(cache, weights_only=True)

codes = data["codes"].cuda()
starts = data["starts"].cuda()
codec = FrozenCodec(torch.load(OUT / "codec.pt", weights_only=True)).cuda()
if data["codes"].shape[1:] != (7, 16, 16):
    raise ValueError("Wrong code bank shape for this 2D model")
if "eigenvalues" in data and not torch.allclose(
    data["eigenvalues"].cpu(),
    torch.load(OUT / "codec.pt", weights_only=True)["eigenvalues"].cpu(),
):
    raise ValueError("Code bank and codec use different latent bases")
val = {
    k: v.float().cuda()
    for k, v in torch.load(args.validation_pairs, weights_only=True)[
        "validation"
    ].items()
}
val_coords = F.avg_pool2d(val["anatomy"][:, 2:5], 4)
model = make_model().cuda()
ema = copy.deepcopy(model).eval().requires_grad_(False)
optimizer = torch.optim.AdamW(
    model.parameters(), lr=3e-4, weight_decay=1e-4, fused=True
)
scaler = torch.amp.GradScaler("cuda", init_scale=1024)
begin_step = 0
best = float("inf")
history = []
if args.resume:
    ck = torch.load(args.resume, weights_only=True)
    model.load_state_dict(ck["online"])
    ema.load_state_dict(ck["model"])
    optimizer.load_state_dict(ck["optimizer"])
    scaler.load_state_dict(ck["scaler"])
    begin_step = ck["step"]
    best = ck["best"]
    history = ck["history"]
    for group in optimizer.param_groups:
        group["fused"] = True
        group["foreach"] = False
    for state in optimizer.state.values():
        if torch.is_tensor(state.get("step")):
            state["step"] = state["step"].cuda()
    if "cuda_rng" in ck:
        torch.cuda.set_rng_state(ck["cuda_rng"].cpu())
    if "torch_rng" in ck:
        torch.set_rng_state(ck["torch_rng"].cpu())
train_forward = torch.compile(model, mode="reduce-overhead") if args.compile else model
ema_params = list(ema.parameters())
model_params = list(model.parameters())
log(
    f"PRIOR parameters={sum(p.numel() for p in model.parameters())} codes={len(codes)} batch={args.batch} start={begin_step} target_steps={args.steps}"
)
(OUT / "settings.json").write_text(
    json.dumps(
        dict(
            model_config=MODEL_CONFIG,
            steps=args.steps,
            batch=args.batch,
            subjects=int(data["subjects"]),
            source="independent unit Gaussian",
            objective="complete nonlesional MRI-code RF velocity MSE",
            synthetic_mask_conditioning=False,
            decoder_frozen=True,
        ),
        indent=2,
    )
)


def metrics(p):
    t = val["target"]
    mask = val["mask"]
    mm = mask[:, :, 3:-3, 3:-3]
    mae = ((p - t).abs() * mask).flatten(1).sum(1) / mask.flatten(1).sum(1).clamp_min(1)
    ss = compute_ssim_and_cs(p, t, 2, (7, 7), (1.5, 1.5), data_range=2)[0]
    ss = (ss * mm).flatten(1).sum(1) / mm.flatten(1).sum(1).clamp_min(1)
    assert torch.isfinite(p).all() and torch.equal(p[mask == 0], t[mask == 0])
    return dict(masked_mae=float(mae.mean()), masked_ssim=float(ss.mean()))


def checkpoint(step):
    return dict(
        model=cpu_state(ema),
        online=cpu_state(model),
        optimizer=optimizer.state_dict(),
        scaler=scaler.state_dict(),
        model_config=MODEL_CONFIG,
        step=step,
        best=best,
        history=history,
        torch_rng=torch.get_rng_state(),
        cuda_rng=torch.cuda.get_rng_state(),
    )


for step in range(begin_step + 1, args.steps + 1):
    idx = torch.randint(len(codes), (args.batch,), device="cuda")
    target = codes[idx].float()
    coords = coordinate_maps(starts[idx])
    noise = torch.randn_like(target)
    timepoints = torch.rand(len(target), device="cuda")
    tv = timepoints[:, None, None, None]
    state = (1 - tv) * noise + tv * target
    optimizer.zero_grad(set_to_none=True)
    lr = (
        3e-4
        * min(1, step / 300)
        * (0.15 + 0.85 * 0.5 * (1 + math.cos(math.pi * min(step / args.steps, 1))))
    )
    for group in optimizer.param_groups:
        group["lr"] = lr
    velocity = predict_velocity(train_forward, state, coords, timepoints)
    loss = (velocity - target + noise).square().mean()
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1, foreach=True)
    scale = scaler.get_scale()
    scaler.step(optimizer)
    scaler.update()
    if scaler.get_scale() >= scale:
        with torch.no_grad():
            decay = min(0.999, (step + 1) / (step + 10))
            torch._foreach_lerp_(ema_params, model_params, 1 - decay)
    if step % 200 == 0:
        elapsed = time.time() - START
        rate = (step - begin_step) / elapsed
        log(
            f"TRAIN step={step} mse={float(loss):.5f} seconds={elapsed:.0f} steps_per_second={rate:.2f} remaining_minutes={(args.steps - step) / max(rate, 1e-9) / 60:.1f}"
        )
    if step % args.validate_every == 0 or step == args.steps:
        row = dict(step=step, validation={})
        with torch.inference_mode():
            for gamma in [0.25, 0.5, 1.0]:
                p = inpaint(
                    ema,
                    codec,
                    val["image"],
                    val["mask"],
                    val_coords,
                    steps=32,
                    projection=gamma,
                )
                row["validation"][str(gamma)] = metrics(p)
            choice = min(
                row["validation"],
                key=lambda k: (
                    row["validation"][k]["masked_mae"]
                    + 0.25 * (1 - row["validation"][k]["masked_ssim"])
                ),
            )
            row["projection"] = float(choice)
            chosen = row["validation"][choice]
            score = chosen["masked_mae"] + 0.25 * (1 - chosen["masked_ssim"])
            z = healthy_sample(ema, val_coords[:4], steps=40)
            healthy = codec.decode(z).clamp(-1, 1).cpu()
            torch.save(
                dict(
                    images=healthy,
                    coords=val_coords[:4].cpu(),
                    reference=val["target"][:4].cpu(),
                    label="Unconditional healthy-prior samples at these MNI locations; not patient-matched inpainting",
                ),
                OUT / f"prior_samples_{step:06d}.pt",
            )
        history.append(row)
        if score < best:
            best = score
            bestck = checkpoint(step)
            bestck["projection"] = float(choice)
            atomic_save(bestck, OUT / "best.pt")
        ck = checkpoint(step)
        ck["projection"] = float(choice)
        atomic_save(ck, OUT / "last.pt")
        atomic_save(ck, OUT / f"step_{step:06d}.pt")
        (OUT / "history.json").write_text(json.dumps(history, indent=2))
        log("VALIDATION " + json.dumps(row))
log("COMPLETE")
