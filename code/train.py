"""Training-only selection of the complete multiscale heat-value model."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

import model_core as core
from predict import PREDICTION_COLUMN, predict_from_arrays

warnings.filterwarnings("ignore")

# 模型搜索空间与复现随机种子。
ALL_PAIRS = ((5, 2), (7, 3), (9, 4), (11, 5), (13, 6), (15, 7), (17, 8))
WEIGHT_PENALTIES = np.asarray((0.0, 1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1), float)
GEOMETRY_CANDIDATES = (3, 5, 7, 10)
CORRECTION_CANDIDATES = (0.0, 0.35, 0.70, 1.0)
SELECTION_SEED = 20260827


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rmse(y: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(prediction) - np.asarray(y)) ** 2)))


def stratified_splits(
    coal_types: np.ndarray, seed: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    counts = pd.Series(coal_types).value_counts()
    # 最多使用5折，并受最小煤种样本数限制，避免某一折缺少煤种。
    folds = int(min(5, counts.min()))
    if folds < 2:
        raise ValueError("At least two training batches per coal type are required")
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    dummy = np.zeros(len(coal_types), dtype=float)
    return [(train, valid) for train, valid in splitter.split(dummy, coal_types)]


def within_coal_quartiles(values: np.ndarray, coal_types: np.ndarray) -> np.ndarray:
    # 每个煤种内部固定划分为4个有序环境，用于检验时间/光谱迁移稳定性。
    groups = np.zeros(len(values), dtype=int)
    for coal_type in np.unique(coal_types):
        rows = np.flatnonzero(coal_types == coal_type)
        order = rows[np.argsort(values[rows], kind="mergesort")]
        groups[order] = np.minimum(3, (4 * np.arange(len(order))) // len(order))
    return groups


def training_environments(
    train_ids: np.ndarray, coal_types: np.ndarray, spectra: np.ndarray
) -> list[tuple[str, int, np.ndarray, np.ndarray]]:
    def date_value(batch_id: str) -> int:
        match = re.search(r"(\d+)月(\d+)日", str(batch_id))
        if match is None:
            raise ValueError(f"Batch identifier has no month/day: {batch_id}")
        return 100 * int(match.group(1)) + int(match.group(2))

    time_values = np.asarray([date_value(value) for value in train_ids], float)
    scaled = StandardScaler().fit_transform(np.asarray(spectra, float))
    # 这里的10维PCA只用于构造训练验证环境，不是最终残差模型维数。
    components = min(10, len(scaled) - 1, scaled.shape[1])
    scores = PCA(n_components=components, svd_solver="full").fit_transform(scaled)
    pc1_values = scores[:, 0]
    radial_values = np.zeros(len(scores), dtype=float)
    for coal_type in np.unique(coal_types):
        rows = np.flatnonzero(coal_types == coal_type)
        center = scores[rows].mean(axis=0)
        radial_values[rows] = np.linalg.norm(scores[rows] - center, axis=1)
    # 三类训练环境分别描述时间漂移、主光谱方向和煤种内离群半径。
    families = {
        "time": within_coal_quartiles(time_values, coal_types),
        "pc1": within_coal_quartiles(pc1_values, coal_types),
        "radial": within_coal_quartiles(radial_values, coal_types),
    }
    environments = []
    for family, assignment in families.items():
        for fold in range(4):
            valid_rows = np.flatnonzero(assignment == fold)
            train_rows = np.flatnonzero(assignment != fold)
            environments.append((family, fold + 1, train_rows, valid_rows))
    return environments


def stratified_environments(
    coal_types: np.ndarray,
) -> list[tuple[str, int, np.ndarray, np.ndarray]]:
    return [
        ("coal_stratified", fold, train_rows, valid_rows)
        for fold, (train_rows, valid_rows) in enumerate(
            stratified_splits(coal_types, SELECTION_SEED), start=1
        )
    ]


def fit_convex_weights(
    predictions: np.ndarray, y: np.ndarray, penalty: float
) -> np.ndarray:
    matrix = np.asarray(predictions, float)
    target = np.asarray(y, float)
    count = matrix.shape[1]
    initial = np.full(count, 1.0 / count, dtype=float)
    target_variance = max(float(np.var(target)), 1.0)

    def objective(weights: np.ndarray) -> float:
        error = matrix @ weights - target
        return float(np.mean(error**2) / target_variance + penalty * np.sum(weights**2))

    result = minimize(
        objective,
        initial,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * count,
        constraints={"type": "eq", "fun": lambda weights: float(weights.sum() - 1.0)},
        # 优化器收敛设置。
        options={"ftol": 1.0e-12, "maxiter": 1000},
    )
    if not result.success:
        raise RuntimeError(f"Convex weight fitting failed: {result.message}")
    weights = np.maximum(np.asarray(result.x, float), 0.0)
    return weights / weights.sum()


def select_penalty(
    predictions: np.ndarray,
    y: np.ndarray,
    coal_types: np.ndarray,
    seed: int,
) -> tuple[float, pd.DataFrame]:
    rows: list[dict[str, float]] = []
    splits = stratified_splits(coal_types, seed)
    for penalty in WEIGHT_PENALTIES:
        fold_scores = []
        for train_rows, valid_rows in splits:
            weights = fit_convex_weights(
                predictions[train_rows], y[train_rows], float(penalty)
            )
            fold_scores.append(rmse(y[valid_rows], predictions[valid_rows] @ weights))
        rows.append(
            {
                "penalty": float(penalty),
                "mean_rmse": float(np.mean(fold_scores)),
                "std_rmse": float(np.std(fold_scores, ddof=1)),
                "worst_rmse": float(np.max(fold_scores)),
            }
        )
    table = pd.DataFrame(rows).sort_values(
        ["mean_rmse", "worst_rmse", "penalty"], kind="mergesort"
    )
    return float(table.iloc[0]["penalty"]), table


def train_all_components(prepared_path: Path, output_path: Path) -> Path:
    started = time.perf_counter()
    with np.load(prepared_path, allow_pickle=False) as data:
        x_train_raw = data["x_train"]
        x_test_raw = data["x_test"]
        y = data["y"]
        groups = data["train_ids"].astype(str)
    arrays: dict[str, np.ndarray] = {}
    with threadpool_limits(limits=1):
        for pair_index, (window, radius) in enumerate(ALL_PAIRS, start=1):
            x_train = core.preprocess(x_train_raw, window)
            x_test = core.preprocess(x_test_raw, window)
            for kernel_index, kernel_name in enumerate(core.KERNEL_NAMES, start=1):
                print(
                    f"[components] pair={pair_index}/{len(ALL_PAIRS)} "
                    f"kernel={kernel_index}/{len(core.KERNEL_NAMES)} {kernel_name}",
                    flush=True,
                )
                arrays[f"oof_{window}_{radius}_{kernel_name}"] = core.fit_oof_kernel(
                    x_train, y, groups, kernel_name, radius
                )
                test_prediction, masks, kernel_thetas = core.fit_test_kernel_with_state(
                    x_train, y, x_test, kernel_name, radius
                )
                arrays[f"test_{window}_{radius}_{kernel_name}"] = test_prediction
                for aggregate, theta in enumerate(kernel_thetas):
                    arrays[
                        f"theta_{window}_{radius}_{kernel_name}_{aggregate}"
                    ] = theta
                if kernel_index == 1:
                    for aggregate, mask in enumerate(masks):
                        arrays[f"mask_{window}_{radius}_{aggregate}"] = mask
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **arrays)
    print(f"[components] seconds={time.perf_counter() - started:.1f}", flush=True)
    return output_path


def component_matrix(
    cache: dict[str, np.ndarray], pair: tuple[int, int], split: str
) -> np.ndarray:
    window, radius = pair
    return np.column_stack(
        [cache[f"{split}_{window}_{radius}_{name}"] for name in core.KERNEL_NAMES]
    )


def apply_hierarchical_weights(
    cache: dict[str, np.ndarray], kernel_weights: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    pair_oof = []
    pair_test = []
    for pair, weights in zip(ALL_PAIRS, kernel_weights):
        pair_oof.append(component_matrix(cache, pair, "oof") @ weights)
        pair_test.append(component_matrix(cache, pair, "test") @ weights)
    return np.column_stack(pair_oof), np.column_stack(pair_test)


def flattened_components(
    cache: dict[str, np.ndarray], split: str
) -> np.ndarray:
    return np.column_stack(
        [
            cache[f"{split}_{window}_{radius}_{kernel_name}"]
            for window, radius in ALL_PAIRS
            for kernel_name in core.KERNEL_NAMES
        ]
    )


def fit_minimax_environment_weights(
    predictions: np.ndarray,
    y: np.ndarray,
    environments: list[tuple[str, int, np.ndarray, np.ndarray]],
) -> np.ndarray:
    matrix = np.asarray(predictions, float)
    target = np.asarray(y, float)
    count = matrix.shape[1]
    variance = max(float(np.var(target)), 1.0)
    initial_weights = np.full(count, 1.0 / count, dtype=float)
    initial_losses = [
        float(np.mean((matrix[valid] @ initial_weights - target[valid]) ** 2) / variance)
        for _, _, _, valid in environments
    ]
    initial = np.r_[initial_weights, max(initial_losses)]

    def objective(parameters: np.ndarray) -> float:
        return float(parameters[-1] + 1.0e-8 * np.sum(parameters[:-1] ** 2))

    constraints: list[dict[str, object]] = [
        {
            "type": "eq",
            "fun": lambda parameters: float(parameters[:-1].sum() - 1.0),
        }
    ]
    for _, _, _, valid_rows in environments:
        constraints.append(
            {
                "type": "ineq",
                "fun": lambda parameters, rows=valid_rows: float(
                    parameters[-1]
                    - np.mean((matrix[rows] @ parameters[:-1] - target[rows]) ** 2)
                    / variance
                ),
            }
        )
    result = minimize(
        objective,
        initial,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * count + [(0.0, None)],
        constraints=constraints,
        # Minimax约束较多，允许更多迭代。
        options={"ftol": 1.0e-12, "maxiter": 3000},
    )
    if not result.success:
        raise RuntimeError(f"Minimax environment fitting failed: {result.message}")
    weights = np.maximum(np.asarray(result.x[:-1], float), 0.0)
    return weights / weights.sum()


def fit_hierarchical_weights(
    cache: dict[str, np.ndarray],
    y: np.ndarray,
    coal_types: np.ndarray,
    rows: np.ndarray | None = None,
    seed: int = SELECTION_SEED,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    selected_rows = np.arange(len(y)) if rows is None else np.asarray(rows, int)
    kernel_weights = []
    kernel_penalties = []
    pair_train = []
    pair_test = []
    kernel_curves: dict[str, list[dict[str, float]]] = {}
    for pair_index, pair in enumerate(ALL_PAIRS):
        train_matrix = component_matrix(cache, pair, "oof")
        test_matrix = component_matrix(cache, pair, "test")
        penalty, curve = select_penalty(
            train_matrix[selected_rows],
            y[selected_rows],
            coal_types[selected_rows],
            seed + 101 * pair_index,
        )
        weights = fit_convex_weights(
            train_matrix[selected_rows], y[selected_rows], penalty
        )
        kernel_weights.append(weights)
        kernel_penalties.append(penalty)
        pair_train.append(train_matrix @ weights)
        pair_test.append(test_matrix @ weights)
        kernel_curves[f"{pair[0]}_{pair[1]}"] = curve.to_dict(orient="records")
    pair_train_matrix = np.column_stack(pair_train)
    pair_test_matrix = np.column_stack(pair_test)
    pair_penalty, pair_curve = select_penalty(
        pair_train_matrix[selected_rows],
        y[selected_rows],
        coal_types[selected_rows],
        seed + 5000,
    )
    pair_weights = fit_convex_weights(
        pair_train_matrix[selected_rows], y[selected_rows], pair_penalty
    )
    return (
        np.asarray(kernel_weights),
        pair_weights,
        pair_train_matrix,
        pair_test_matrix,
        {
            "kernel_penalties": kernel_penalties,
            "pair_penalty": pair_penalty,
            "kernel_penalty_curves": kernel_curves,
            "pair_penalty_curve": pair_curve.to_dict(orient="records"),
        },
    )


def crossfit_base_prediction(
    cache: dict[str, np.ndarray],
    y: np.ndarray,
    coal_types: np.ndarray,
    environments: list[tuple[str, int, np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, list[dict[str, object]]]:
    prediction_sum = np.zeros(len(y), dtype=float)
    prediction_count = np.zeros(len(y), dtype=int)
    audits: list[dict[str, object]] = []
    for task_index, (family, fold, train_rows, valid_rows) in enumerate(
        environments, start=1
    ):
        kernel_weights, pair_weights, pair_train, _, metadata = fit_hierarchical_weights(
            cache,
            y,
            coal_types,
            rows=train_rows,
            seed=SELECTION_SEED + 10000 * task_index,
        )
        prediction_sum[valid_rows] += pair_train[valid_rows] @ pair_weights
        prediction_count[valid_rows] += 1
        audits.append(
            {
                "family": family,
                "fold": fold,
                "train_rows": train_rows.tolist(),
                "valid_rows": valid_rows.tolist(),
                "kernel_weights": kernel_weights.tolist(),
                "pair_weights": pair_weights.tolist(),
                "pair_penalty": metadata["pair_penalty"],
            }
        )
    family_count = len({family for family, _, _, _ in environments})
    if not np.all(prediction_count == family_count):
        raise RuntimeError("Every row must be validated once in each environment family")
    prediction = prediction_sum / prediction_count
    if not np.isfinite(prediction).all():
        raise RuntimeError("Cross-fitted base prediction is incomplete")
    return prediction, audits


def select_residual_configuration(
    base_crossfit: np.ndarray,
    y: np.ndarray,
    coal_types: np.ndarray,
    spectra: np.ndarray,
    wavelengths: np.ndarray,
    environments: list[tuple[str, int, np.ndarray, np.ndarray]],
) -> tuple[int, float, pd.DataFrame]:
    rows: list[dict[str, float | int]] = []
    fold_common = []
    for family, fold, train_rows, valid_rows in environments:
        excluded, _ = core.c_mad3_mask(
            base_crossfit[train_rows], y[train_rows], coal_types[train_rows]
        )
        parameters = core.fit_calibration(
            base_crossfit[train_rows],
            y[train_rows],
            coal_types[train_rows],
            excluded,
        )
        calibrated_train = core.apply_calibration(
            base_crossfit[train_rows], coal_types[train_rows], parameters
        )
        calibrated_valid = core.apply_calibration(
            base_crossfit[valid_rows], coal_types[valid_rows], parameters
        )
        train_scores, valid_scores = core.geometry_scores(
            spectra[train_rows], spectra[valid_rows], wavelengths
        )
        fold_common.append(
            (
                fold,
                family,
                train_rows,
                valid_rows,
                calibrated_train,
                calibrated_valid,
                train_scores,
                valid_scores,
            )
        )
    for components in GEOMETRY_CANDIDATES:
        for scale in CORRECTION_CANDIDATES:
            fold_scores = []
            for (
                _,
                _,
                train_rows,
                valid_rows,
                calibrated_train,
                calibrated_valid,
                train_scores,
                valid_scores,
            ) in fold_common:
                arrays = {
                    "coal_train": coal_types[train_rows],
                    "coal_test": coal_types[valid_rows],
                    "raw_test_calibrated": calibrated_valid,
                    "geometry_train_scores": train_scores,
                    "geometry_test_scores": valid_scores,
                    "training_residuals": y[train_rows] - calibrated_train,
                    "geometry_components": np.asarray(components),
                    "correction_scale": np.asarray(scale),
                }
                fold_scores.append(rmse(y[valid_rows], predict_from_arrays(arrays)))
            rows.append(
                {
                    "geometry_components": components,
                    "correction_scale": scale,
                    "mean_rmse": float(np.mean(fold_scores)),
                    "std_rmse": float(np.std(fold_scores, ddof=1)),
                    "worst_rmse": float(np.max(fold_scores)),
                    **{f"fold_{index + 1}_rmse": score for index, score in enumerate(fold_scores)},
                }
            )
    table = pd.DataFrame(rows)
    # 先以scale=0构造无残差修正基线；候选必须同时不恶化平均折和最差折。
    baseline = table.loc[table["correction_scale"].eq(0.0)].sort_values(
        ["mean_rmse", "worst_rmse", "geometry_components"], kind="mergesort"
    ).iloc[0]
    eligible = table.loc[
        table["mean_rmse"].le(float(baseline["mean_rmse"]) + 1.0e-12)
        & table["worst_rmse"].le(float(baseline["worst_rmse"]) + 1.0e-12)
    ]
    # 在合格候选中按平均RMSE、最差折RMSE、较小修正和较低维数依次决胜。
    selected = eligible.sort_values(
        ["mean_rmse", "worst_rmse", "correction_scale", "geometry_components"],
        kind="mergesort",
    ).iloc[0]
    table["selected"] = (
        table["geometry_components"].eq(int(selected["geometry_components"]))
        & table["correction_scale"].eq(float(selected["correction_scale"]))
    )
    return (
        int(selected["geometry_components"]),
        float(selected["correction_scale"]),
        table,
    )


def write_final_candidate(
    xfdata_root: Path,
    user_data_root: Path,
    prediction_output: Path,
    prepared_path: Path,
    component_path: Path,
    weight_estimator: str,
) -> tuple[Path, Path]:
    started = time.perf_counter()
    with np.load(prepared_path, allow_pickle=False) as loaded:
        data = {key: np.asarray(loaded[key]) for key in loaded.files}
    with np.load(component_path, allow_pickle=False) as loaded:
        cache = {key: np.asarray(loaded[key]) for key in loaded.files}
    y = data["y"].astype(float)
    coal_train = data["coal_train"].astype(str)
    coal_test = data["coal_test"].astype(str)
    all_training_kernel_weights, all_training_pair_weights, _, _, weight_metadata = (
        fit_hierarchical_weights(cache, y, coal_train)
    )
    if weight_estimator == "stratified_bagged":
        environments = stratified_environments(coal_train)
    else:
        environments = training_environments(
            data["train_ids"].astype(str), coal_train, data["x_train"]
        )
    base_crossfit, crossfit_audit = crossfit_base_prediction(
        cache, y, coal_train, environments
    )
    component_oof = flattened_components(cache, "oof")
    component_test = flattened_components(cache, "test")
    if weight_estimator == "environment_minimax":
        flat_weights = fit_minimax_environment_weights(
            component_oof, y, environments
        )
        weight_matrix = flat_weights.reshape(len(ALL_PAIRS), len(core.KERNEL_NAMES))
        pair_weights = weight_matrix.sum(axis=1)
        kernel_weights = np.divide(
            weight_matrix,
            pair_weights[:, None],
            out=np.zeros_like(weight_matrix),
            where=pair_weights[:, None] > 1.0e-12,
        )
        estimator_description = "nonnegative minimax loss across twelve training environments"
    else:
        kernel_weights = np.mean(
            np.asarray([item["kernel_weights"] for item in crossfit_audit], float), axis=0
        )
        kernel_weights /= kernel_weights.sum(axis=1, keepdims=True)
        pair_weights = np.mean(
            np.asarray([item["pair_weights"] for item in crossfit_audit], float), axis=0
        )
        pair_weights /= pair_weights.sum()
        weight_matrix = kernel_weights * pair_weights[:, None]
        flat_weights = weight_matrix.ravel()
        estimator_description = "mean of outer-training-fold weight vectors"
    raw_oof = component_oof @ flat_weights
    raw_test = component_test @ flat_weights
    resampled_wavelengths = np.linspace(
        float(data["wavelengths"].min()), float(data["wavelengths"].max()), core.RAW_POINTS
    )
    resampled_train = np.vstack(
        [np.interp(resampled_wavelengths, data["wavelengths"], row) for row in data["x_train"]]
    ).astype(np.float32)
    components, correction_scale, residual_table = select_residual_configuration(
        base_crossfit,
        y,
        coal_train,
        resampled_train,
        resampled_wavelengths,
        environments,
    )
    excluded, cutoff = core.c_mad3_mask(raw_oof, y, coal_train)
    calibration = core.fit_calibration(raw_oof, y, coal_train, excluded)
    calibrated_train = core.apply_calibration(raw_oof, coal_train, calibration)
    calibrated_test = core.apply_calibration(raw_test, coal_test, calibration)
    train_scores, test_scores = core.geometry_scores(
        data["x_train"], data["x_test"], data["wavelengths"]
    )

    sample = pd.read_csv(core.submission_template_path(xfdata_root), encoding="utf-8-sig")
    sample_ids = sample.iloc[:, 0].astype(str).to_numpy()
    test_ids = data["test_ids"].astype(str)
    positions = {batch_id: row for row, batch_id in enumerate(test_ids)}
    order = np.asarray([positions[batch_id] for batch_id in sample_ids], dtype=int)
    calibration_types = np.asarray(sorted(calibration), dtype=str)
    # 仅固化训练参数；推理时从xfdata重建测试预测。
    arrays: dict[str, np.ndarray] = {
        "model_format_version": np.asarray(2, dtype=np.int32),
        "train_ids": np.asarray(data["train_ids"], dtype=str),
        "coal_train": np.asarray(coal_train, dtype=str),
        "wavelengths": np.asarray(data["wavelengths"], dtype=np.float32),
        "training_residuals": y - calibrated_train,
        "geometry_components": np.asarray(components),
        "correction_scale": np.asarray(correction_scale),
        "calibration_coal_types": calibration_types,
        "calibration_slopes": np.asarray(
            [calibration[key][0] for key in calibration_types], dtype=np.float64
        ),
        "calibration_intercepts": np.asarray(
            [calibration[key][1] for key in calibration_types], dtype=np.float64
        ),
        "kernel_weights": kernel_weights,
        "pair_weights": pair_weights,
        "component_weights": flat_weights,
    }
    for key, value in cache.items():
        if key.startswith("mask_") or key.startswith("theta_"):
            arrays[key] = value
    final_prediction = predict_from_arrays(
        {
            "coal_train": coal_train,
            "coal_test": coal_test[order],
            "raw_test_calibrated": calibrated_test[order],
            "geometry_train_scores": train_scores,
            "geometry_test_scores": test_scores[order],
            "training_residuals": arrays["training_residuals"],
            "geometry_components": arrays["geometry_components"],
            "correction_scale": arrays["correction_scale"],
        },
        apply_frozen_offset=False,
    )
    model_dir = user_data_root / "model_data"
    analysis_dir = user_data_root / "tmp_data" / "analysis"
    model_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir.mkdir(parents=True, exist_ok=True)
    weights_path = model_dir / "model_weights.npz"
    np.savez_compressed(weights_path, **arrays)
    submission_path = prediction_output
    submission_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {sample.columns[0]: sample_ids, PREDICTION_COLUMN: final_prediction}
    ).to_csv(
        submission_path,
        index=False,
        encoding="utf-8",
        float_format="%.10f",
        lineterminator="\n",
    )
    residual_table.to_csv(analysis_dir / "residual_selection.csv", index=False, encoding="utf-8")
    pd.DataFrame(
        {
            "row": np.arange(len(y)),
            "coal_type": coal_train,
            "target": y,
            "crossfit_base_prediction": base_crossfit,
            "error": base_crossfit - y,
        }
    ).to_csv(analysis_dir / "base_crossfit_predictions.csv", index=False, encoding="utf-8")
    metadata = {
        "method": "training_selected_multiscale_multikernel_residual_model",
        "model_format_version": 2,
        "contains_test_specific_arrays": False,
        "training_rows": int(len(y)),
        "test_rows": int(len(test_ids)),
        "candidate_pairs": [list(pair) for pair in ALL_PAIRS],
        "kernel_names": list(core.KERNEL_NAMES),
        "kernel_weights": kernel_weights.tolist(),
        "pair_weights": pair_weights.tolist(),
        "all_training_kernel_weights_for_audit": all_training_kernel_weights.tolist(),
        "all_training_pair_weights_for_audit": all_training_pair_weights.tolist(),
        "weight_estimator": weight_estimator,
        "final_weight_estimator": estimator_description,
        "weight_selection": weight_metadata,
        "crossfit_weight_audit": crossfit_audit,
        "validation_environment_families": sorted(
            {family for family, _, _, _ in environments}
        ),
        "geometry_components": components,
        "correction_scale": correction_scale,
        "c_mad3_cutoff": float(cutoff),
        "excluded_training_rows": np.flatnonzero(excluded).astype(int).tolist(),
        "calibration_parameters": {key: list(value) for key, value in calibration.items()},
        "base_crossfit_rmse": rmse(y, base_crossfit),
        "final_oof_rmse": rmse(y, calibrated_train),
        "component_predictions_sha256": file_sha256(component_path),
        "weights_sha256": file_sha256(weights_path),
        "submission_sha256": file_sha256(submission_path),
        "seconds": float(time.perf_counter() - started),
    }
    (model_dir / "model_config.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"[selection] components={components}, scale={correction_scale:.2f}, "
        f"base_crossfit_rmse={metadata['base_crossfit_rmse']:.6f}, "
        f"seconds={metadata['seconds']:.1f}",
        flush=True,
    )
    return weights_path, submission_path


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True, encoding="utf-8")
    parser = argparse.ArgumentParser()
    package_root = Path(__file__).resolve().parents[1]
    parser.add_argument(
        "--xfdata-root", type=Path, default=package_root / "xfdata"
    )
    parser.add_argument(
        "--user-data-root", type=Path, default=package_root / "user_data"
    )
    parser.add_argument(
        "--prediction-output",
        type=Path,
        default=package_root / "prediction_result" / "result",
    )
    parser.add_argument("--stage", choices=("data", "components", "final", "all"), default="all")
    parser.add_argument(
        "--weight-estimator",
        choices=("stratified_bagged", "environment_bagged", "environment_minimax"),
        # 默认外层折构造策略。
        default="stratified_bagged",
    )
    args = parser.parse_args()
    xfdata_root = args.xfdata_root.resolve()
    user_data_root = args.user_data_root.resolve()
    prediction_output = args.prediction_output.resolve()
    artifact_dir = user_data_root / "tmp_data"
    prepared_path = artifact_dir / "prepared_data.npz"
    component_path = artifact_dir / "component_predictions.npz"
    if args.stage in {"data", "all"}:
        prepared_path = core.prepare_data(xfdata_root, artifact_dir)
    if args.stage in {"components", "all"}:
        if not prepared_path.exists():
            raise FileNotFoundError("Run the data stage first")
        component_path = train_all_components(prepared_path, component_path)
    if args.stage in {"final", "all"}:
        if not prepared_path.exists() or not component_path.exists():
            raise FileNotFoundError("Run the data and component stages first")
        # 固定线性代数线程数，避免PCA/SVD因线程调度产生微小浮点差异。
        with threadpool_limits(limits=1):
            write_final_candidate(
                xfdata_root,
                user_data_root,
                prediction_output,
                prepared_path,
                component_path,
                args.weight_estimator,
            )
if __name__ == "__main__":
    main()
