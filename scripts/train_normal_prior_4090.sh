#!/usr/bin/env bash
# Keep each host's models, codec and cache inside its own explicit run directory.
set -euo pipefail
if [[ $# -lt 1 ]]; then
  echo "Usage: bash scripts/train_normal_prior_4090.sh RUN_DIRECTORY [--preflight | --resume | --stage STAGE ...]" >&2
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
if [[ ${1:-} == --preflight ]]; then
  shift
  set -- --stage preflight "$@"
fi
exec python -m strokefill3d.normal_prior.train --config configs/atlas3_normal_prior_4090.yaml \
  --manifest "$data_root/manifests/atlas3_train.csv" \
  --mask-manifest "$data_root/manifests/atlas3_masks_train.csv" \
  --val-manifest "$data_root/manifests/anatomical_v1/atlas3_val.csv" \
  --test-manifest "$data_root/manifests/anatomical_v1/atlas3_test.csv" \
  --output "$run_dir" --device cuda "$@"
