#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

mkdir -p "$PROJECT_ROOT/prediction_result"
"$PYTHON_BIN" "$PROJECT_ROOT/code/predict.py" \
  --xfdata-root "$PROJECT_ROOT/xfdata" \
  --weights "$PROJECT_ROOT/user_data/model_data/model_weights.npz" \
  --output "$PROJECT_ROOT/prediction_result/result"
