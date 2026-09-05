"""Validate the frozen model and submission schema."""
from __future__ import annotations

import argparse
import hashlib
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from predict import IDENTIFIER_COLUMN, PREDICTION_COLUMN, predict_from_arrays


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    package_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-root", type=Path, default=package_root)
    parser.add_argument("--archive", type=Path)
    args = parser.parse_args()
    root = args.package_root.resolve()
    submission_path = root / "submit.csv"
    weights_path = root / "model" / "model_weights.npz"

    raw = submission_path.read_bytes()
    if raw.startswith(b"\xef\xbb\xbf"):
        raise ValueError("submit.csv must use UTF-8 without a byte-order mark")
    raw.decode("utf-8", errors="strict")
    submission = pd.read_csv(submission_path, encoding="utf-8")
    if submission.columns.tolist() != [IDENTIFIER_COLUMN, PREDICTION_COLUMN]:
        raise ValueError(f"Unexpected submission columns: {submission.columns.tolist()}")
    if submission.isna().any().any():
        raise ValueError("Submission contains missing values")
    if not submission[IDENTIFIER_COLUMN].is_unique:
        raise ValueError("Submission identifiers are not unique")
    if not np.isfinite(submission[PREDICTION_COLUMN].to_numpy(float)).all():
        raise ValueError("Submission contains non-finite predictions")

    with np.load(weights_path, allow_pickle=False) as arrays:
        prediction = predict_from_arrays(arrays)
        batch_ids = np.asarray(arrays["batch_ids"]).astype(str)
    if not np.array_equal(submission[IDENTIFIER_COLUMN].astype(str).to_numpy(), batch_ids):
        raise ValueError("Submission row order does not match the frozen model")
    if not np.allclose(
        submission[PREDICTION_COLUMN].to_numpy(float), prediction, rtol=0.0, atol=5.0e-9
    ):
        raise ValueError("Submission predictions do not match the frozen model")

    if args.archive is not None:
        required = {
            "submit.csv",
            "README.md",
            "requirements.txt",
            "model/model_weights.npz",
            "model/model_config.json",
            "model/base_component_predictions.npz",
            "src/predict.py",
            "src/train_full.py",
            "src/train_and_freeze.py",
            "src/verify_package.py",
        }
        with zipfile.ZipFile(args.archive.resolve()) as archive:
            entries = {name.replace("\\", "/") for name in archive.namelist()}
        missing = required - entries
        if missing:
            raise ValueError(f"Archive is missing required files: {sorted(missing)}")
    print(
        {
            "rows": len(submission),
            "columns": submission.columns.tolist(),
            "submit_sha256": file_sha256(submission_path),
            "weights_sha256": file_sha256(weights_path),
        }
    )


if __name__ == "__main__":
    main()
