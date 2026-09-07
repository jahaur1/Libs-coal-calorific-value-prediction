"""Inference entry point for the organizer-compatible frozen model."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import model_core as core


IDENTIFIER_COLUMN = "名称"
PREDICTION_COLUMN = "预测发热量_MJ_KG"
ALL_PAIRS = ((5, 2), (7, 3), (9, 4), (11, 5), (13, 6), (15, 7), (17, 8))


def robust_scale(values: np.ndarray) -> float:
    values = np.asarray(values, float)
    mad = float(1.4826 * np.median(np.abs(values - np.median(values))))
    return max(mad, float(np.median(np.abs(values))), 1.0e-12)


def pairwise_distances(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    differences = left[:, None, :] - right[None, :, :]
    return np.sqrt(np.sum(differences**2, axis=2))


def predict_from_arrays(
    arrays: Mapping[str, np.ndarray], *, apply_frozen_offset: bool = False
) -> np.ndarray:
    """Apply the training-selected residual correction to base predictions."""
    if apply_frozen_offset:
        raise ValueError("Frozen output offsets are not supported by model format v2")
    geometry_components = int(np.asarray(arrays["geometry_components"]).item())
    correction_scale = float(np.asarray(arrays["correction_scale"]).item())
    coal_train = np.asarray(arrays["coal_train"]).astype(str)
    coal_test = np.asarray(arrays["coal_test"]).astype(str)
    train_scores = np.asarray(arrays["geometry_train_scores"], float)[
        :, :geometry_components
    ]
    test_scores = np.asarray(arrays["geometry_test_scores"], float)[
        :, :geometry_components
    ]
    residuals = np.asarray(arrays["training_residuals"], float)
    raw_test = np.asarray(arrays["raw_test_calibrated"], float)
    correction = np.zeros(len(test_scores), float)

    for coal_type in np.unique(coal_test):
        train_rows = np.flatnonzero(coal_train == coal_type)
        test_rows = np.flatnonzero(coal_test == coal_type)
        if len(train_rows) < 2:
            continue
        within = pairwise_distances(train_scores[train_rows], train_scores[train_rows])
        np.fill_diagonal(within, np.inf)
        support_radius = max(float(np.median(np.min(within, axis=1))), 1.0e-12)
        query_distances = pairwise_distances(
            test_scores[test_rows], train_scores[train_rows]
        )
        neighbor_count = max(1, int(np.ceil(np.sqrt(len(train_rows)))))
        coal_residuals = residuals[train_rows]
        clip = robust_scale(coal_residuals)
        for local_row, test_row in enumerate(test_rows):
            order = np.argsort(
                query_distances[local_row], kind="mergesort"
            )[:neighbor_count]
            nearest_distance = float(query_distances[local_row, order[0]])
            confidence = support_radius / (support_radius + nearest_distance)
            local_signal = float(np.median(coal_residuals[order]))
            correction[test_row] = confidence * np.clip(local_signal, -clip, clip)

    for coal_type in np.unique(coal_test):
        mask = coal_test == coal_type
        correction[mask] -= float(np.mean(correction[mask]))
    return raw_test + correction_scale * correction


def calibration_from_weights(
    arrays: Mapping[str, np.ndarray],
) -> dict[str, tuple[float, float]]:
    coal_types = np.asarray(arrays["calibration_coal_types"]).astype(str)
    slopes = np.asarray(arrays["calibration_slopes"], float)
    intercepts = np.asarray(arrays["calibration_intercepts"], float)
    if not (len(coal_types) == len(slopes) == len(intercepts)):
        raise ValueError("Calibration parameter lengths do not match")
    return {
        coal_type: (float(slope), float(intercept))
        for coal_type, slope, intercept in zip(coal_types, slopes, intercepts)
    }


def validate_training_snapshot(
    data: Mapping[str, np.ndarray], arrays: Mapping[str, np.ndarray]
) -> None:
    version = int(np.asarray(arrays["model_format_version"]).item())
    if version != 2:
        raise ValueError(f"Unsupported model format version: {version}")
    for key in ("train_ids", "coal_train"):
        actual = np.asarray(data[key]).astype(str)
        expected = np.asarray(arrays[key]).astype(str)
        if not np.array_equal(actual, expected):
            raise ValueError(f"Organizer training data does not match frozen {key}")
    actual_wavelengths = np.asarray(data["wavelengths"], dtype=np.float32)
    expected_wavelengths = np.asarray(arrays["wavelengths"], dtype=np.float32)
    if not np.array_equal(actual_wavelengths, expected_wavelengths):
        raise ValueError("Organizer wavelength grid does not match the trained model")


def base_prediction_from_frozen_model(
    data: Mapping[str, np.ndarray], arrays: Mapping[str, np.ndarray]
) -> np.ndarray:
    x_train_raw = np.asarray(data["x_train"])
    x_test_raw = np.asarray(data["x_test"])
    y = np.asarray(data["y"])
    component_predictions: list[np.ndarray] = []
    for window, radius in ALL_PAIRS:
        x_train = core.preprocess(x_train_raw, window)
        x_test = core.preprocess(x_test_raw, window)
        masks = [
            np.asarray(arrays[f"mask_{window}_{radius}_{aggregate}"], bool)
            for aggregate in range(core.MASK_AGGREGATES)
        ]
        for kernel_name in core.KERNEL_NAMES:
            thetas = [
                np.asarray(
                    arrays[f"theta_{window}_{radius}_{kernel_name}_{aggregate}"],
                    float,
                )
                for aggregate in range(core.MASK_AGGREGATES)
            ]
            component_predictions.append(
                core.predict_test_kernel_from_state(
                    x_train,
                    y,
                    x_test,
                    kernel_name,
                    radius,
                    masks,
                    thetas,
                )
            )
    matrix = np.column_stack(component_predictions)
    weights = np.asarray(arrays["component_weights"], float)
    if matrix.shape[1] != len(weights):
        raise ValueError("Frozen component weights do not match the base models")
    return matrix @ weights


def predict_from_xfdata(
    xfdata_root: Path, weights_path: Path
) -> tuple[np.ndarray, np.ndarray, str]:
    data = core.load_competition_arrays(xfdata_root)
    with np.load(weights_path, allow_pickle=False) as loaded:
        arrays = {key: np.asarray(loaded[key]) for key in loaded.files}
    validate_training_snapshot(data, arrays)
    with threadpool_limits(limits=1):
        raw_test = base_prediction_from_frozen_model(data, arrays)
        calibration = calibration_from_weights(arrays)
        calibrated_test = core.apply_calibration(
            raw_test, np.asarray(data["coal_test"]).astype(str), calibration
        )
        train_scores, test_scores = core.geometry_scores(
            data["x_train"], data["x_test"], data["wavelengths"]
        )

    sample = pd.read_csv(core.submission_template_path(xfdata_root), encoding="utf-8-sig")
    sample_ids = sample.iloc[:, 0].astype(str).to_numpy()
    test_ids = np.asarray(data["test_ids"]).astype(str)
    positions = {batch_id: row for row, batch_id in enumerate(test_ids)}
    if len(positions) != len(test_ids):
        raise ValueError("Test batch identifiers are not unique")
    try:
        order = np.asarray([positions[batch_id] for batch_id in sample_ids], dtype=int)
    except KeyError as error:
        raise ValueError(
            f"Submission template contains an unknown batch: {error.args[0]}"
        ) from error
    if len(order) != len(test_ids) or len(np.unique(order)) != len(order):
        raise ValueError("Submission template and test batches do not have a one-to-one match")
    prediction = predict_from_arrays(
        {
            "coal_train": arrays["coal_train"],
            "coal_test": np.asarray(data["coal_test"]).astype(str)[order],
            "raw_test_calibrated": calibrated_test[order],
            "geometry_train_scores": train_scores,
            "geometry_test_scores": test_scores[order],
            "training_residuals": arrays["training_residuals"],
            "geometry_components": arrays["geometry_components"],
            "correction_scale": arrays["correction_scale"],
        }
    )
    return sample_ids, prediction, str(sample.columns[0])


def write_submission(
    xfdata_root: Path, weights_path: Path, output_path: Path
) -> Path:
    batch_ids, prediction, identifier_column = predict_from_xfdata(
        xfdata_root, weights_path
    )
    submission = pd.DataFrame(
        {identifier_column: batch_ids, PREDICTION_COLUMN: prediction}
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(
        output_path,
        index=False,
        encoding="utf-8",
        float_format="%.10f",
        lineterminator="\n",
    )
    return output_path


def main() -> None:
    package_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--xfdata-root", type=Path, default=package_root / "xfdata"
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=package_root / "user_data" / "model_data" / "model_weights.npz",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=package_root / "prediction_result" / "result",
    )
    args = parser.parse_args()
    path = write_submission(
        args.xfdata_root.resolve(), args.weights.resolve(), args.output.resolve()
    )
    print(path)


if __name__ == "__main__":
    main()
