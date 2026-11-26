"""One real 96-cube update per stage plus source gradients and NIfTI roundtrip."""

import time
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from .core import (
    VERSION,
    Codec,
    amp,
    coordinate_maps,
    file_hash,
    fit_source,
    load_bundle,
    make_autoencoder,
    make_prior,
    prior_config,
    velocity,
)
from .data import NormalPatches, fixed_cases
from .io import atomic_json, atomic_save, cpu_state


def run(cfg, args, out, device):
    from .infer import generate_nifti
    from .train import ae_step, optimizer_for, restore_optimizer

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    start = time.time()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    print("PREFLIGHT: preparing real clean 3D target and cortical mask", flush=True)
    dataset = NormalPatches(
        args.manifest,
        cfg["patch_size"],
        cfg["seed"],
        cfg["data"]["exclusion_margin"],
        1,
    )
    val = NormalPatches(
        args.val_manifest,
        cfg["patch_size"],
        cfg["seed"] + 1,
        cfg["data"]["exclusion_margin"],
        1,
    )
    cases = fixed_cases(val, 1, args.mask_manifest, cfg["seed"] + 2)
    case = cases[0]
    target = dataset[0]["target"][None].to(device)
    print("PREFLIGHT: 96-cube AE forward/backward and checkpoint resume", flush=True)
    ae = make_autoencoder(cfg["autoencoder"]["model"]).to(device)
    opt = optimizer_for(ae, cfg["autoencoder"]["lr"], device)
    scale = torch.amp.GradScaler(
        device.type, enabled=device.type == "cuda", init_scale=1024
    )
    loss = ae_step(ae, target, 1, cfg["autoencoder"])
    scale.scale(loss).backward()
    scale.unscale_(opt)
    torch.nn.utils.clip_grad_norm_(ae.parameters(), 1)
    scale.step(opt)
    scale.update()
    ae_loss = float(loss)
    atomic_save(
        {"model": cpu_state(ae), "optimizer": opt.state_dict()}, out / "ae_roundtrip.pt"
    )
    ae2 = make_autoencoder(cfg["autoencoder"]["model"]).to(device)
    saved = torch.load(out / "ae_roundtrip.pt", map_location="cpu", weights_only=True)
    ae2.load_state_dict(saved["model"])
    opt2 = optimizer_for(ae2, cfg["autoencoder"]["lr"], device)
    restore_optimizer(opt2, saved["optimizer"], device)
    opt2.zero_grad(set_to_none=True)
    loss2 = ae_step(ae2, target, 2, cfg["autoencoder"])
    scale.scale(loss2).backward()
    scale.unscale_(opt2)
    scale.step(opt2)
    scale.update()
    del loss2
    del opt, opt2, ae
    ae2.eval().requires_grad_(False)
    with torch.no_grad(), amp(device):
        mu, _ = ae2.encode(target)
    points = mu.float().cpu().movedim(1, 0).flatten(1).double()
    mean = points.mean(1)
    std = points.std(1).clamp_min(0.001)
    c = len(mean)
    codec_saved = {
        "kind": VERSION + "_codec",
        "ae_config": cfg["autoencoder"]["model"],
        "model": cpu_state(ae2),
        "patch_size": cfg["patch_size"],
        "factor": 2 ** (len(cfg["autoencoder"]["model"]["channels"]) - 1),
        "mean": mean.float()[None, :, None, None, None],
        "std": std.float()[None, :, None, None, None],
        "whiten": torch.eye(c),
        "unwhiten": torch.eye(c),
        "quality": {"passed": False, "functional_test_only": True},
    }
    atomic_save(codec_saved, out / "codec.pt")
    del ae2, mu, points, target
    codec = Codec(
        torch.load(out / "codec.pt", map_location="cpu", weights_only=True)
    ).to(device)
    target = case["target"][None].to(device)
    code = codec.encode(target)
    atomic_save(
        {"codes": code.half().cpu(), "codec_sha256": file_hash(out / "codec.pt")},
        out / "code_shard.pt",
    )
    loaded = torch.load(out / "code_shard.pt", weights_only=True)
    assert loaded["codec_sha256"] == file_hash(out / "codec.pt")
    print("PREFLIGHT: full-profile RF forward/backward on 48-cube codes", flush=True)
    mc = prior_config(cfg["prior"]["model"], codec.channels)
    model = make_prior(mc).to(device)
    opt = optimizer_for(model, cfg["prior"]["lr"], device)
    inference_coords = coordinate_maps(
        case["origin"][None].to(device), code.shape[2:], codec.factor
    )
    code = code.repeat(cfg["prior"]["batch_size"], 1, 1, 1, 1)
    coords = inference_coords.repeat(len(code), 1, 1, 1, 1)
    noise = torch.randn_like(code)
    t = torch.rand(len(code), device=device)
    tv = t[:, None, None, None, None]
    x = (1 - tv) * noise + tv * code
    v = velocity(model, x, coords, t)
    loss = (v - code + noise).square().mean()
    scale = torch.amp.GradScaler(
        device.type, enabled=device.type == "cuda", init_scale=1024
    )
    scale.scale(loss).backward()
    scale.unscale_(opt)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
    scale.step(opt)
    scale.update()
    rf_loss = float(loss)
    (out / "prior").mkdir(exist_ok=True)
    atomic_save(
        {
            "kind": VERSION + "_prior",
            "model": cpu_state(model),
            "online": cpu_state(model),
            "optimizer": opt.state_dict(),
            "model_config": mc,
            "codec_sha256": file_hash(out / "codec.pt"),
            "step": 1,
        },
        out / "prior/best.pt",
    )
    del opt, model, noise, v, loss, x, code
    torch.cuda.empty_cache()
    try:
        load_bundle(out, device=device)
    except ValueError as e:
        if "reconstruction gate" not in str(e):
            raise
    else:
        raise RuntimeError("Unqualified test codec was accepted for production")
    print(
        "PREFLIGHT: differentiating through the frozen RF with observed pixels only",
        flush=True,
    )
    model, codec, _ = load_bundle(out, device=device, allow_functional=True)
    image = case["image"][None].to(device)
    mask = case["mask"][None].to(device)
    pred, _, _, history = fit_source(
        model,
        codec,
        image,
        mask,
        inference_coords,
        steps=2,
        iterations=1,
        checkpoint_steps=True,
    )
    assert torch.isfinite(pred).all() and torch.equal(
        pred[mask == 0], target[mask == 0]
    )
    # Exercise complete-volume normalization, ROI planning and NIfTI output.
    print("PREFLIGHT: NIfTI inference roundtrip and outside-mask identity", flush=True)
    source = nib.load(case["source"])
    raw = np.asarray(source.dataobj, dtype=np.float32)
    fullmask = np.zeros(raw.shape, np.uint8)
    starts = case["starts"].numpy()
    sl = tuple(slice(int(s), int(s + n)) for s, n in zip(starts, cfg["patch_size"]))
    fullmask[sl] = case["mask"][0].numpy().astype(np.uint8)
    nib.save(nib.Nifti1Image(fullmask, source.affine), out / "synthetic_mask.nii.gz")
    generated = generate_nifti(
        model,
        codec,
        case["source"],
        out / "synthetic_mask.nii.gz",
        out / "functional_inference.nii.gz",
        device,
        {
            "seed": 909,
            "flow_steps": 1,
            "iterations": 0,
            "trust_weight": 0.001,
            "boundary_weight": 4.0,
            "dilation": 0,
            "margin": 8,
        },
        {"functional_test_only": True},
    )
    assert np.array_equal(generated[fullmask == 0], raw[fullmask == 0])
    result = {
        "functional_test_only": True,
        "quality_demonstrated": False,
        "patch_size": cfg["patch_size"],
        "latent_shape": list(coords.shape[2:]),
        "rf_batch_size": cfg["prior"]["batch_size"],
        "ae_loss": ae_loss,
        "rf_loss": rf_loss,
        "finite": True,
        "outside_max_error": 0.0,
        "unqualified_codec_rejected": True,
        "source_gradient_test": history,
        "seconds": time.time() - start,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3
        if device.type == "cuda"
        else None,
    }
    atomic_json(result, out / "result.json")
    print(result, flush=True)
