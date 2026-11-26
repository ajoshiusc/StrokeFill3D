#!/usr/bin/env bash
# Explicit run directory plus a process lock prevents shared-disk run collisions.
set -euo pipefail
if [[ $# -lt 1 ]]; then
  echo "Usage: bash scripts/train_anatomical_4090.sh RUN_DIRECTORY [--preflight | --resume CHECKPOINT]" >&2
  exit 2
fi
run_dir=$1
shift
project_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$project_dir"
data_root=${STROKEFILL_DATA_ROOT:-/deneb_disk/data/stroke_inpainting}
python scripts/split_validation_sites.py --holdout "$data_root/manifests/atlas3_holdout.csv" \
  --output-dir "$data_root/manifests/anatomical_v1"
export PYTHONPATH="$project_dir/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
exec python -m strokefill3d.train --config configs/atlas3_anatomical_4090.yaml \
  --manifest "$data_root/manifests/atlas3_train.csv" \
  --mask-manifest "$data_root/manifests/atlas3_masks_train.csv" \
  --val-manifest "$data_root/manifests/anatomical_v1/atlas3_val.csv" \
  --test-manifest "$data_root/manifests/anatomical_v1/atlas3_test.csv" \
  --output "$run_dir" --device cuda "$@"
