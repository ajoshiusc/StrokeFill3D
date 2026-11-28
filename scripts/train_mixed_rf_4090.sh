#!/usr/bin/env bash
# One explicit local run directory per host; data may be shared read-only.
set -euo pipefail
if [[ $# -lt 1 ]]; then
  echo 'Usage: bash scripts/train_mixed_rf_4090.sh RUN_DIRECTORY [--preflight | --resume | --evaluate-test]' >&2
  exit 2
fi
run_dir=$1
shift
project_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$project_dir"
export PYTHONPATH="$project_dir/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
exec python -m strokefill3d.mixed_rf.train --config configs/mixed_rf_4090.yaml \
  --data "${STROKEFILL_MIXED_DATA:-/deneb_disk/data/stroke_inpainting/mixed_rf_v2}" \
  --output "$run_dir" --device cuda "$@"
