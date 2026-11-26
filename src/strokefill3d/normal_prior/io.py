import fcntl
import json
import os
import socket
from pathlib import Path

import torch

from .core import VERSION, file_hash


def atomic_save(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(value, tmp)
    os.replace(tmp, path)


def atomic_json(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2))
    os.replace(tmp, path)


class RunLock:
    def __init__(self, out):
        self.path = Path(out) / ".normal_prior.lock"

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = self.path.open("a+")
        try:
            fcntl.flock(self.file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.file.close()
            raise RuntimeError("Another process owns this run directory")
        self.file.seek(0)
        self.file.truncate()
        self.file.write(f"{socket.gethostname()} pid={os.getpid()}\n")
        self.file.flush()
        return self

    def __exit__(self, *args):
        self.file.close()


def initialize_run(out, cfg, paths, resume):
    path = Path(out) / "run.json"
    identity = {
        "version": VERSION,
        "config": cfg,
        "data_hashes": {k: file_hash(v) for k, v in paths.items() if v},
    }
    if path.exists():
        if not resume:
            raise ValueError(
                "Run already exists. Use --resume to continue it, or a new output directory"
            )
        old = json.loads(path.read_text())
        if any(old[k] != identity[k] for k in ["version", "config", "data_hashes"]):
            raise ValueError(
                "Config/manifests differ from this run; use a new run directory"
            )
    else:
        if resume:
            raise ValueError("No run identity exists to resume")
        existing = [
            p
            for p in Path(out).iterdir()
            if p.name not in ["logs", ".normal_prior.lock", "preflight"]
        ]
        if existing:
            raise ValueError(
                "Refusing to mix this pipeline with a nonempty/unrelated output directory"
            )
        identity.update(
            host=socket.gethostname(), paths={k: str(v) for k, v in paths.items() if v}
        )
        atomic_json(identity, path)
    return identity


def cpu_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def rng_state():
    return {
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
    }


def restore_rng(saved):
    if "torch_rng" in saved:
        torch.set_rng_state(saved["torch_rng"].cpu())
    if torch.cuda.is_available() and saved.get("cuda_rng") is not None:
        torch.cuda.set_rng_state(saved["cuda_rng"].cpu())
