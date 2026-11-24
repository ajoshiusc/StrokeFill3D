"""Train a MONAI conditional rectified-flow model on synthetic inpainting pairs."""
from __future__ import annotations
import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml
from .data import HealthyT1Dataset
from .model import RectifiedFlowUNet


def masked_velocity_mse(pred, target, mask):
    return ((pred.float() - target.float()).square() * mask).sum() / mask.sum().clamp_min(1.0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/train.yaml")
    p.add_argument("--manifest", required=True)
    p.add_argument("--mask-manifest", default=None, help="CSV with a lesion_mask column; donor masks are transferred into healthy-image patches")
    p.add_argument("--output", default="checkpoints")
    p.add_argument("--resume", default=None, help="Resume model and optimizer state from a training checkpoint")
    args = p.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = HealthyT1Dataset(args.manifest, tuple(cfg["patch_size"]), cfg["samples_per_scan"],
                               tuple(cfg["spacing_mm"]), args.mask_manifest)
    loader = DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=True,
                        num_workers=cfg["num_workers"], pin_memory=device.type == "cuda")
    model_cfg = {k: cfg[k] for k in ("channels", "attention_levels", "num_head_channels", "num_res_blocks")}
    model = RectifiedFlowUNet(**model_cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and cfg["amp"])
    start_epoch = 0
    if args.resume:
        state = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        if state.get("scaler"):
            scaler.load_state_dict(state["scaler"])
        step = int(state.get("step", 0))
        # Re-run the checkpoint's current epoch from its beginning, which keeps
        # the epoch schedule simple while retaining learned weights and optimizer.
        start_epoch = max(0, int(state.get("epoch", 1)) - 1)
    else:
        step = 0
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)

    def save_checkpoint(epoch):
        state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                 "scaler": scaler.state_dict(), "epoch": epoch, "step": step, "model_config": model_cfg,
                 "training_config": cfg, "method": "conditional_rectified_flow",
                 "mask_manifest": args.mask_manifest}
        temporary = out / "last.pt.tmp"
        torch.save(state, temporary)
        temporary.replace(out / "last.pt")

    model.train()
    for epoch in range(start_epoch, cfg["epochs"]):
        running = 0.0
        bar = tqdm(loader, desc=f"epoch {epoch+1}/{cfg['epochs']}")
        for batch in bar:
            context = batch["image"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            noise = torch.randn_like(target)
            # Straight interpolation from noise (t=0) to healthy tissue (t=1).
            t = torch.rand((target.shape[0],), device=device)
            shape = (target.shape[0], 1, 1, 1, 1)
            t_view = t.view(shape)
            interpolant = (1.0 - t_view) * noise + t_view * target
            sample = torch.where(mask > 0, interpolant, context)
            velocity_target = target - noise
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16,
                                enabled=device.type == "cuda" and cfg["amp"]):
                velocity = model(sample, context, mask, t * cfg["time_embedding_scale"])
                loss = masked_velocity_mse(velocity, velocity_target, mask)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            scaler.step(optimizer); scaler.update()
            running += float(loss.detach()); step += 1
            bar.set_postfix(loss=f"{float(loss.detach()):.4f}")
            if cfg.get("checkpoint_every_steps", 0) and step % cfg["checkpoint_every_steps"] == 0:
                save_checkpoint(epoch + 1)
        save_checkpoint(epoch + 1)
        print(f"epoch={epoch+1} mean_masked_velocity_mse={running/max(1,len(loader)):.6f}")

if __name__ == "__main__": main()
