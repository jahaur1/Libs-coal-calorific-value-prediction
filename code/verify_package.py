"""Static and output checks for the organizer submission package."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import model_core as core
from predict import PREDICTION_COLUMN


REQUIRED_PATHS = (
    "README.md",
    "requirements.txt",
    "xfdata",
    "user_data/model_data/model_weights.npz",
    "user_data/model_data/model_config.json",
    "user_data/tmp_data",
    "prediction_result/result",
    "code/model_core.py",
    "code/train.py",
    "code/predict.py",
    "code/verify_package.py",
    "train.sh",
    "test.sh",
)
FORBIDDEN_MODEL_KEYS = {
    "batch_ids",
    "coal_test",
    "raw_test_calibrated",
    "geometry_test_scores",
    "expected_prediction",
    "frozen_output_offset",
}


def verify(root: Path, xfdata_root: Path | None = None) -> None:
    root = root.resolve()
    missing = [relative for relative in REQUIRED_PATHS if not (root / relative).exists()]
    if missing:
        raise FileNotFoundError(f"Missing required package paths: {missing}")

    result_path = root / "prediction_result" / "result"
    raw = result_path.read_bytes()
    if raw.startswith(b"\xef\xbb\xbf"):
        raise ValueError("prediction_result/result must be UTF-8 without BOM")
    raw.decode("utf-8")
    result = pd.read_csv(result_path, encoding="utf-8")
    if len(result.columns) != 2 or result.columns[1] != PREDICTION_COLUMN:
        raise ValueError(
            f"Expected identifier plus {PREDICTION_COLUMN}; found {result.columns.tolist()}"
        )
    if result.iloc[:, 0].astype(str).duplicated().any():
        raise ValueError("Prediction identifiers are not unique")
    prediction = pd.to_numeric(result[PREDICTION_COLUMN], errors="coerce").to_numpy()
    if not np.isfinite(prediction).all():
        raise ValueError("Predictions contain missing or non-finite values")

    weights_path = root / "user_data" / "model_data" / "model_weights.npz"
    with np.load(weights_path, allow_pickle=False) as arrays:
        present = FORBIDDEN_MODEL_KEYS.intersection(arrays.files)
        if present:
            raise ValueError(f"Model contains test-specific output keys: {sorted(present)}")
        if int(np.asarray(arrays["model_format_version"]).item()) != 2:
            raise ValueError("Expected model format version 2")

    if xfdata_root is not None:
        sample = pd.read_csv(
            core.submission_template_path(xfdata_root), encoding="utf-8-sig"
        )
        if not sample.iloc[:, 0].astype(str).equals(result.iloc[:, 0].astype(str)):
            raise ValueError("Prediction identifiers do not match the submission template")
    print(
        f"verified: rows={len(result)}, columns={result.columns.tolist()}, "
        "utf8=true, forbidden_model_keys=[]"
    )


def main() -> None:
    package_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=package_root)
    parser.add_argument("--xfdata-root", type=Path)
    args = parser.parse_args()
    verify(args.root, args.xfdata_root.resolve() if args.xfdata_root else None)


if __name__ == "__main__":
    main()
