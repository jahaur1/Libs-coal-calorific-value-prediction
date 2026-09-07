#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

"$PYTHON_BIN" "$PROJECT_ROOT/code/train.py" \
  --xfdata-root "$PROJECT_ROOT/xfdata" \
  --user-data-root "$PROJECT_ROOT/user_data" \
  --prediction-output "$PROJECT_ROOT/prediction_result/result" \
  --stage all
