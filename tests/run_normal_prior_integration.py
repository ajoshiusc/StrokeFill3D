"""Private functional test only; temporary weights are never usable as research results."""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import torch
import yaml

from strokefill3d.normal_prior import train
from strokefill3d.normal_prior.core import file_hash

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--result", type=Path)
args_cli = parser.parse_args()
repo = Path(__file__).resolve().parents[1]
cfg = yaml.safe_load((repo / "configs/atlas3_normal_prior_4090.yaml").read_text())
cfg["patch_size"] = [48] * 3
cfg["data"]["workers"] = cfg["cache"]["workers"] = 0
cfg["calibration_samples"] = 2
cfg["autoencoder"].update(steps=1, validate_every=1, save_every=1)
cfg["autoencoder"]["model"].update(
    channels=[8, 16], num_res_blocks=1, latent_channels=4
)
cfg["cache"].update(samples=4, shard_size=2)
cfg["prior"].update(
    steps=1, validate_every=1, save_every=1, batch_size=1, accumulation=1
)
cfg["prior"]["model"].update(
    channels=[8, 16, 32, 32],
    num_res_blocks=1,
    attention_levels=[False, False, False, True],
)
cfg["validation"]["count"] = 1
for key in ("quick", "full"):
    cfg["validation"][key].update(cases=1, seeds=[909], flow_steps=1, iterations=0)
base = (
    Path(os.environ.get("STROKEFILL_DATA_ROOT", "/deneb_disk/data/stroke_inpainting"))
    / "manifests"
)
with tempfile.TemporaryDirectory(prefix="normal3d_private_functional_") as folder:
    folder = Path(folder)
    config = folder / "config.yaml"
    config.write_text(yaml.safe_dump(cfg))
    out = folder / "run"
    args = [
        "functional-test",
        "--config",
        str(config),
        "--manifest",
        str(base / "atlas3_train.csv"),
        "--val-manifest",
        str(base / "anatomical_v1/atlas3_val.csv"),
        "--test-manifest",
        str(base / "anatomical_v1/atlas3_test.csv"),
        "--mask-manifest",
        str(base / "atlas3_masks_train.csv"),
        "--output",
        str(out),
        "--device",
        "cuda",
    ]
    # An untrained AE cannot pass the real gate. Bypass ONLY in this temporary
    # private test to exercise cache/trainer orchestration; production is unmodified.
    with (
        patch.object(train, "quality_passed", return_value=True),
        patch.object(sys, "argv", args),
    ):
        train.main()
    sha = file_hash(out / "codec.pt")
    before = torch.load(out / "prior/last.pt", map_location="cpu", weights_only=True)[
        "online"
    ]
    with patch.object(
        sys, "argv", args + ["--stage", "prior", "--resume", "--max-steps", "2"]
    ):
        train.main()
    ck = torch.load(out / "prior/last.pt", map_location="cpu", weights_only=True)
    assert ck["step"] == 2 and ck["codec_sha256"] == sha
    assert any(not torch.equal(v, ck["online"][k]) for k, v in before.items()), (
        "Resume did not update RF weights"
    )
    assert file_hash(out / "codec.pt") == sha
    assert (out / "prior/final_full_validation/metrics.json").is_file()
    result = {
        "functional_test_only": True,
        "quality_demonstrated": False,
        "all_stages_completed": True,
        "resumed_rf_step": ck["step"],
        "codec_fingerprint_preserved": True,
        "temporary_gate_bypass": True,
        "patch_size": cfg["patch_size"],
    }
    if args_cli.result:
        args_cli.result.parent.mkdir(parents=True, exist_ok=True)
        args_cli.result.write_text(json.dumps(result, indent=2))
    print("INTEGRATION_OK", json.dumps(result), flush=True)
