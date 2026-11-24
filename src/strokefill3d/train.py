"""Training entry point for synthetic-mask T1 reconstruction."""
from __future__ import annotations
import argparse
from pathlib import Path
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm
from .data import HealthyT1Dataset
from .model import StrokeFillNet


def masked_l1(pred, target, mask):
    inside = (torch.abs(pred - target) * mask).sum() / mask.sum().clamp_min(1.0)
    outside = (torch.abs(pred - target) * (1 - mask)).mean()
    return inside + 0.05 * outside


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/train.yaml")
    p.add_argument("--manifest", required=True)
    p.add_argument("--output", default="checkpoints")
    args = p.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = HealthyT1Dataset(args.manifest, tuple(cfg["patch_size"]), cfg["samples_per_scan"], tuple(cfg["spacing_mm"]))
    loader = DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=True,
                        num_workers=cfg["num_workers"], pin_memory=device.type == "cuda")
    model = StrokeFillNet(tuple(cfg["patch_size"]), cfg["feature_size"], cfg["use_checkpoint"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and cfg["amp"])
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    step = 0
    model.train()
    for epoch in range(cfg["epochs"]):
        running = 0.0
        bar = tqdm(loader, desc=f"epoch {epoch+1}/{cfg['epochs']}")
        for batch in bar:
            image, mask, target = (batch[k].to(device, non_blocking=True) for k in ("image", "mask", "target"))
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda" and cfg["amp"]):
                pred = model(image, mask)
                loss = masked_l1(pred, target, mask)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            scaler.step(optimizer); scaler.update()
            running += float(loss.detach()); step += 1
            bar.set_postfix(loss=f"{float(loss.detach()):.4f}")
        state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch + 1,
                 "step": step, "config": cfg}
        torch.save(state, out / "last.pt")
        print(f"epoch={epoch+1} mean_loss={running/max(1,len(loader)):.6f}")

if __name__ == "__main__": main()
