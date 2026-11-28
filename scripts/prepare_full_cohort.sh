#!/usr/bin/env bash
# Resume preparation and publish readiness only after the subject-level audit.
set -euo pipefail
project_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$project_dir"
export PYTHONPATH="$project_dir/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
: "${STROKEFILL_SYNTHSTRIP_WEIGHTS:?Set path to synthstrip.1.pt}"
: "${STROKEFILL_TEMPLATE:?Set path to canonical MNI T1}"
: "${STROKEFILL_TEMPLATE_MASK:?Set path to template brain mask}"
data_dir=${STROKEFILL_MIXED_DATA:-/deneb_disk/data/stroke_inpainting/mixed_rf_v2}
if [[ ! -f "$data_dir/dataset.json" ]]; then
  python scripts/prepare_mixed_cohort.py --output "$data_dir" \
    --synthstrip-weights "$STROKEFILL_SYNTHSTRIP_WEIGHTS" \
    --template "$STROKEFILL_TEMPLATE" --template-mask "$STROKEFILL_TEMPLATE_MASK" "$@"
fi
python scripts/audit_mixed_cohort.py --data "$data_dir"
