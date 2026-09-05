"""Portable inference for the frozen heat-value model."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd


IDENTIFIER_COLUMN = "名称"
PREDICTION_COLUMN = "预测发热量_MJ_KG"
# 最终参赛方案的统一残差收缩系数。它控制残差专家对基础预测的影响强度，
# 不随测试批次或测试样本编号变化。
CORRECTION_SCALE = 0.70

# 只使用训练阶段生成的前五个标准化几何分量计算同煤种邻域距离，
# 以限制小样本下距离估计的维度和方差。
GEOMETRY_COMPONENTS = 5


def robust_scale(values: np.ndarray) -> float:
    values = np.asarray(values, float)
    mad = float(1.4826 * np.median(np.abs(values - np.median(values))))
    return max(mad, float(np.median(np.abs(values))), 1.0e-12)


def pairwise_distances(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    differences = left[:, None, :] - right[None, :, :]
    return np.sqrt(np.sum(differences**2, axis=2))


def predict_from_arrays(
    arrays: Mapping[str, np.ndarray], *, apply_frozen_offset: bool = True
) -> np.ndarray:
    geometry_components = int(
        np.asarray(arrays.get("geometry_components", GEOMETRY_COMPONENTS)).item()
    )
    correction_scale = float(
        np.asarray(arrays.get("correction_scale", CORRECTION_SCALE)).item()
    )
    coal_train = np.asarray(arrays["coal_train"]).astype(str)
    coal_test = np.asarray(arrays["coal_test"]).astype(str)
    train_scores = np.asarray(arrays["geometry_train_scores"], float)[:, :geometry_components]
    test_scores = np.asarray(arrays["geometry_test_scores"], float)[:, :geometry_components]
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
        query_distances = pairwise_distances(test_scores[test_rows], train_scores[train_rows])
        neighbor_count = max(1, int(np.ceil(np.sqrt(len(train_rows)))))
        coal_residuals = residuals[train_rows]
        clip = robust_scale(coal_residuals)
        for local_row, test_row in enumerate(test_rows):
            order = np.argsort(query_distances[local_row], kind="mergesort")[:neighbor_count]
            nearest_distance = float(query_distances[local_row, order[0]])
            confidence = support_radius / (support_radius + nearest_distance)
            local_signal = float(np.median(coal_residuals[order]))
            correction[test_row] = confidence * np.clip(local_signal, -clip, clip)

    for coal_type in np.unique(coal_test):
        mask = coal_test == coal_type
        correction[mask] -= float(np.mean(correction[mask]))
    prediction = raw_test + correction_scale * correction
    if apply_frozen_offset and "frozen_output_offset" in arrays:
        prediction = prediction + np.asarray(arrays["frozen_output_offset"], float)

    if apply_frozen_offset and "expected_prediction" in arrays:
        expected = np.asarray(arrays["expected_prediction"], float)
        if not np.allclose(prediction, expected, rtol=0.0, atol=5.0e-9):
            difference = float(np.max(np.abs(prediction - expected)))
            raise RuntimeError(f"Frozen-weight consistency check failed: max difference={difference}")
    return prediction


def write_submission(weights_path: Path, output_path: Path) -> Path:
    with np.load(weights_path, allow_pickle=False) as arrays:
        batch_ids = np.asarray(arrays["batch_ids"]).astype(str)
        prediction = predict_from_arrays(arrays)
    if len(np.unique(batch_ids)) != len(batch_ids):
        raise ValueError("Batch identifiers are not unique")
    submission = pd.DataFrame(
        {IDENTIFIER_COLUMN: batch_ids, PREDICTION_COLUMN: prediction}
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
        "--weights",
        type=Path,
        default=package_root / "model" / "model_weights.npz",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=package_root / "submit.csv",
    )
    args = parser.parse_args()
    path = write_submission(args.weights.resolve(), args.output.resolve())
    print(path)


if __name__ == "__main__":
    main()
