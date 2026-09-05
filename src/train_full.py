"""Complete fixed training pipeline from the competition raw files."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
import warnings
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    ConstantKernel,
    DotProduct,
    Matern,
    RationalQuadratic,
    RBF,
    WhiteKernel,
)
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from predict import PREDICTION_COLUMN, predict_from_arrays

warnings.filterwarnings("ignore")


# 以下常量是最终参赛方案的全局算法配置，对所有训练和测试批次统一生效。
# 它们控制随机复现、光谱预处理、模型容量和多尺度表示，不包含针对某个测试样本的修补值。
SEED = 0  # 固定波长掩码扰动；GPR 本身没有额外的随机采样步骤。
FFT_CUTOFF = 0.02  # 置零最低频的 2%，用于抑制缓慢变化的光谱背景。
VIP_THRESHOLD = 1.0  # 保留 PLS-VIP 不低于 1 的波长通道。
VIP_COMPONENTS = 20  # 计算 VIP 时的最大 PLS 分量数，实际值受样本数限制。
GPR_RESTARTS = 2  # GPR 核超参数优化的额外重启次数。
MASK_AGGREGATES = 5  # 全训练拟合时对 VIP 掩码做轻微扰动并平均，降低选择方差。
RAW_POINTS = 512  # LVSE/Haar 几何分支使用的统一波长采样点数。

# 每个二元组依次表示 Savitzky-Golay 导数窗口和局部均值半径。
FIXED_PAIRS = ((13, 6), (11, 5), (15, 7))
# 三个尺度分支的历史最终融合权重，和为 1。
PAIR_WEIGHTS = np.asarray(
    (0.34725658675111365, 0.4623235887661649, 0.19041982448272152),
    dtype=float,
)
KERNEL_NAMES = ("rbf", "matern15", "matern25", "rational_quadratic", "dot_product")
# 与 KERNEL_NAMES 一一对应的固定核融合权重，和为 1。
KERNEL_WEIGHTS = np.asarray((0.068, 0.534, 0.254, 0.068, 0.076), dtype=float)
LABEL_COLUMNS = (
    "batch_day",
    "total_moisture",
    "analysis_moisture",
    "ash",
    "target_q",
    "hydrogen",
    "sulfur",
)
# 每项依次为中心谱带、左侧连续谱参考区和右侧连续谱参考区。
# 这些波段用于构建局部连续谱残差与 Haar 多尺度特征。
EMISSION_BANDS = (
    (306.0, 310.0, 302.0, 305.0, 311.0, 314.0),
    (484.0, 488.0, 480.0, 483.0, 489.0, 492.0),
    (653.0, 660.0, 648.0, 652.0, 661.0, 665.0),
    (774.0, 781.0, 768.0, 773.0, 782.0, 788.0),
)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_label_file(path: Path) -> pd.DataFrame:
    frame = pd.read_excel(path, engine="openpyxl").iloc[:, : len(LABEL_COLUMNS)].copy()
    frame.columns = LABEL_COLUMNS
    frame["batch_day"] = frame["batch_day"].astype(str).str.strip()
    frame["coal_type"] = path.stem
    frame["batch_name"] = frame["coal_type"] + frame["batch_day"]
    return frame


def collect_batches(root: Path, split: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if not root.is_dir():
        raise FileNotFoundError(f"Missing data directory: {root}")
    for coal_dir in sorted(root.iterdir()):
        if not coal_dir.is_dir():
            continue
        for batch_dir in sorted(coal_dir.iterdir()):
            if not batch_dir.is_dir():
                continue
            files = sorted(batch_dir.glob("*.csv"))
            if files:
                rows.append(
                    {
                        "split": split,
                        "coal_type": coal_dir.name,
                        "batch_name": batch_dir.name,
                        "batch_dir": str(batch_dir),
                        "spectrum_count": len(files),
                    }
                )
    if not rows:
        raise FileNotFoundError(f"No {split} batches found under {root}")
    return pd.DataFrame(rows)


def build_batch_index(project_root: Path) -> pd.DataFrame:
    data_root = project_root / "data"
    train = collect_batches(data_root / "训练集", "train")
    label_files = sorted((data_root / "训练集标签").glob("*.xlsx"))
    if not label_files:
        raise FileNotFoundError("No training label workbooks were found")
    labels = pd.concat([read_label_file(path) for path in label_files], ignore_index=True)
    columns = ["batch_name", *LABEL_COLUMNS[1:]]
    train = train.merge(labels[columns], on="batch_name", how="left", validate="one_to_one")
    if train["target_q"].isna().any():
        missing = train.loc[train["target_q"].isna(), "batch_name"].tolist()
        raise ValueError(f"Missing labels for training batches: {missing}")
    test = collect_batches(data_root / "测试集", "test")
    for column in LABEL_COLUMNS[1:]:
        test[column] = np.nan
    return pd.concat([train, test], ignore_index=True)


@lru_cache(maxsize=None)
def read_spectrum(path: str) -> tuple[np.ndarray, np.ndarray]:
    frame = pd.read_csv(path, skiprows=4, usecols=[0, 1])
    return (
        frame.iloc[:, 0].to_numpy(np.float32),
        frame.iloc[:, 1].to_numpy(np.float32),
    )


def load_batch_mean(batch_dir: str) -> tuple[np.ndarray, np.ndarray]:
    files = sorted(Path(batch_dir).glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No spectra found under {batch_dir}")
    loaded = [read_spectrum(str(path)) for path in files]
    wavelength = loaded[0][0]
    if any(
        len(item[0]) != len(wavelength)
        or not np.allclose(item[0], wavelength, rtol=0.0, atol=1.0e-6)
        for item in loaded[1:]
    ):
        raise ValueError(f"Wavelength grids differ inside {batch_dir}")
    return wavelength, np.vstack([item[1] for item in loaded]).mean(axis=0)


def prepare_data(project_root: Path, artifact_dir: Path) -> Path:
    started = time.perf_counter()
    index = build_batch_index(project_root)
    means: list[np.ndarray] = []
    reference_wavelengths: np.ndarray | None = None
    for row in index.itertuples(index=False):
        wavelengths, mean = load_batch_mean(row.batch_dir)
        if reference_wavelengths is None:
            reference_wavelengths = wavelengths
        elif len(wavelengths) != len(reference_wavelengths) or not np.allclose(
            wavelengths, reference_wavelengths, rtol=0.0, atol=1.0e-6
        ):
            raise ValueError("Competition batches do not share one wavelength grid")
        means.append(mean)
    if reference_wavelengths is None:
        raise ValueError("No spectra were loaded")
    spectra = np.asarray(means, dtype=np.float32)
    train_mask = index["split"].eq("train").to_numpy()
    test_mask = index["split"].eq("test").to_numpy()
    path = artifact_dir / "prepared_data.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        wavelengths=reference_wavelengths,
        x_train=spectra[train_mask],
        x_test=spectra[test_mask],
        y=index.loc[train_mask, "target_q"].to_numpy(np.float32),
        train_ids=np.asarray(index.loc[train_mask, "batch_name"].astype(str), dtype=str),
        test_ids=np.asarray(index.loc[test_mask, "batch_name"].astype(str), dtype=str),
        coal_train=np.asarray(index.loc[train_mask, "coal_type"].astype(str), dtype=str),
        coal_test=np.asarray(index.loc[test_mask, "coal_type"].astype(str), dtype=str),
    )
    print(
        f"[data] train={int(train_mask.sum())}, test={int(test_mask.sum())}, "
        f"channels={spectra.shape[1]}, seconds={time.perf_counter() - started:.1f}",
        flush=True,
    )
    return path


def fft_highpass(x: np.ndarray) -> np.ndarray:
    output = []
    for signal in x:
        transformed = np.fft.fft(signal)
        keep = int(len(signal) * FFT_CUTOFF)
        transformed[:keep] = 0
        transformed[-keep:] = 0
        output.append(np.real(np.fft.ifft(transformed)))
    return np.asarray(output)


def first_derivative(x: np.ndarray, window: int) -> np.ndarray:
    return np.asarray(
        [savgol_filter(row, window_length=window, polyorder=2, deriv=1) for row in x]
    )


def msc(x: np.ndarray) -> np.ndarray:
    reference = x.mean(axis=0)
    reference_mean = float(reference.mean())
    reference_variance = float(np.var(reference))
    corrected = np.empty_like(x)
    for row, sample in enumerate(x):
        slope = np.dot(sample - sample.mean(), reference - reference_mean) / (
            x.shape[1] * reference_variance
        )
        corrected[row] = (sample - (sample.mean() - slope * reference_mean)) / slope
    return corrected


def preprocess(x: np.ndarray, window: int) -> np.ndarray:
    return msc(first_derivative(fft_highpass(x), window))


def select_vip_mask(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    components = max(1, min(VIP_COMPONENTS, len(x) - 1, x.shape[1]))
    model = PLSRegression(n_components=components, scale=False)
    model.fit(x, y)
    scores = model.x_scores_
    weights = model.x_weights_
    loadings = model.y_loadings_
    explained = np.sum(scores**2, axis=0) * loadings.ravel() ** 2
    denominator = float(explained.sum())
    if denominator <= 0.0:
        return np.ones(x.shape[1], dtype=bool)
    vip = np.sqrt(
        x.shape[1] * np.sum(weights**2 * explained, axis=1) / denominator
    )
    mask = vip >= VIP_THRESHOLD
    if not mask.any():
        mask[int(np.argmax(vip))] = True
    if mask.all():
        mask[int(np.argmin(vip))] = False
    return mask


def local_mean(x: np.ndarray, radius: int) -> np.ndarray:
    output = np.empty_like(x)
    for column in range(x.shape[1]):
        low = max(0, column - radius)
        high = min(x.shape[1], column + radius + 1)
        output[:, column] = x[:, low:high].mean(axis=1)
    return output


def make_gpr(kernel_name: str) -> GaussianProcessRegressor:
    if kernel_name == "rbf":
        kernel = ConstantKernel(1.0) * RBF(10.0) + WhiteKernel(1.0)
    elif kernel_name == "matern15":
        kernel = ConstantKernel(1.0) * Matern(10.0, nu=1.5) + WhiteKernel(1.0)
    elif kernel_name == "matern25":
        kernel = ConstantKernel(1.0) * Matern(10.0, nu=2.5) + WhiteKernel(1.0)
    elif kernel_name == "rational_quadratic":
        kernel = ConstantKernel(1.0) * RationalQuadratic(10.0, alpha=1.0) + WhiteKernel(1.0)
    elif kernel_name == "dot_product":
        kernel = ConstantKernel(1.0) * DotProduct(sigma_0=1.0) + WhiteKernel(1.0)
    else:
        raise KeyError(kernel_name)
    return GaussianProcessRegressor(
        kernel=kernel,
        n_restarts_optimizer=GPR_RESTARTS,
        normalize_y=True,
        random_state=None,
    )


def augment(x: np.ndarray, mask: np.ndarray, radius: int) -> np.ndarray:
    return np.column_stack((x[:, mask], local_mean(x, radius)[:, mask]))


def fit_oof_kernel(
    x: np.ndarray, y: np.ndarray, groups: np.ndarray, kernel_name: str, radius: int
) -> np.ndarray:
    result = np.full(len(y), np.nan, dtype=np.float64)
    splitter = GroupKFold(5)
    for train_rows, valid_rows in splitter.split(x, y, groups):
        np.random.seed(SEED)
        mask = select_vip_mask(x[train_rows], y[train_rows])
        if int(mask.sum()) < 5:
            mask = np.ones(x.shape[1], dtype=bool)
        train_features = augment(x[train_rows], mask, radius)
        valid_features = augment(x[valid_rows], mask, radius)
        scaler = StandardScaler()
        train_scaled = scaler.fit_transform(train_features)
        valid_scaled = scaler.transform(valid_features)
        model = make_gpr(kernel_name)
        model.fit(train_scaled, y[train_rows])
        result[valid_rows] = model.predict(valid_scaled)
    if not np.isfinite(result).all():
        raise RuntimeError("OOF prediction is incomplete")
    return result


def fit_test_kernel(
    x_train: np.ndarray,
    y: np.ndarray,
    x_test: np.ndarray,
    kernel_name: str,
    radius: int,
) -> np.ndarray:
    def fit_once(mask: np.ndarray) -> np.ndarray:
        train_features = augment(x_train, mask, radius)
        test_features = augment(x_test, mask, radius)
        scaler = StandardScaler()
        train_scaled = scaler.fit_transform(train_features)
        test_scaled = scaler.transform(test_features)
        np.random.seed(SEED)
        model = make_gpr(kernel_name)
        model.fit(train_scaled, y)
        return model.predict(test_scaled)

    np.random.seed(SEED)
    base_mask = select_vip_mask(x_train, y)
    if int(base_mask.sum()) < 5:
        base_mask = np.ones(x_train.shape[1], dtype=bool)
    prediction = fit_once(base_mask)
    for aggregate in range(1, MASK_AGGREGATES):
        generator = np.random.RandomState(SEED + 1000 + aggregate)
        mask = base_mask.copy()
        selected = np.flatnonzero(mask)
        mask[selected[generator.rand(len(selected)) < 0.05]] = False
        available = np.flatnonzero(~mask)
        extra_count = int(0.02 * len(mask))
        mask[generator.choice(available, size=extra_count, replace=False)] = True
        prediction += fit_once(mask)
    return prediction / MASK_AGGREGATES


def _train_base_models_single_thread(prepared_path: Path, model_dir: Path) -> Path:
    started = time.perf_counter()
    with np.load(prepared_path, allow_pickle=False) as data:
        x_train_raw = data["x_train"]
        x_test_raw = data["x_test"]
        y = data["y"]
        groups = data["train_ids"].astype(str)
    arrays: dict[str, np.ndarray] = {}
    for pair_index, (window, radius) in enumerate(FIXED_PAIRS, start=1):
        pair_started = time.perf_counter()
        x_train = preprocess(x_train_raw, window)
        x_test = preprocess(x_test_raw, window)
        oof_kernels = []
        test_kernels = []
        for kernel_index, kernel_name in enumerate(KERNEL_NAMES, start=1):
            print(
                f"[base] pair={pair_index}/{len(FIXED_PAIRS)} "
                f"kernel={kernel_index}/{len(KERNEL_NAMES)} {kernel_name}",
                flush=True,
            )
            oof_kernels.append(fit_oof_kernel(x_train, y, groups, kernel_name, radius))
            test_kernels.append(
                fit_test_kernel(x_train, y, x_test, kernel_name, radius)
            )
        arrays[f"oof_{window}_{radius}"] = np.column_stack(oof_kernels) @ KERNEL_WEIGHTS
        arrays[f"test_{window}_{radius}"] = np.column_stack(test_kernels) @ KERNEL_WEIGHTS
        print(
            f"[base] pair=({window},{radius}) seconds={time.perf_counter() - pair_started:.1f}",
            flush=True,
        )
    arrays["raw_oof"] = np.column_stack(
        [arrays[f"oof_{window}_{radius}"] for window, radius in FIXED_PAIRS]
    ) @ PAIR_WEIGHTS
    arrays["raw_test"] = np.column_stack(
        [arrays[f"test_{window}_{radius}"] for window, radius in FIXED_PAIRS]
    ) @ PAIR_WEIGHTS
    path = model_dir / "base_component_predictions.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    print(f"[base] seconds={time.perf_counter() - started:.1f}", flush=True)
    return path


def train_base_models(prepared_path: Path, model_dir: Path) -> Path:
    with threadpool_limits(limits=1):
        return _train_base_models_single_thread(prepared_path, model_dir)


def c_mad3_mask(
    prediction: np.ndarray, y: np.ndarray, coal_types: np.ndarray
) -> tuple[np.ndarray, float]:
    residuals: list[tuple[int, float]] = []
    for coal_type in np.unique(coal_types):
        mask = coal_types == coal_type
        slope, intercept = np.polyfit(prediction[mask], y[mask], 1)
        corrected = slope * prediction[mask] + intercept
        for local, global_row in enumerate(np.flatnonzero(mask)):
            residuals.append((int(global_row), float(y[global_row] - corrected[local])))
    absolute = np.asarray([abs(value) for _, value in residuals], dtype=float)
    median = float(np.median(absolute))
    scaled_mad = float(1.4826 * np.median(np.abs(absolute - median)))
    cutoff = median + 3.0 * scaled_mad
    excluded = np.zeros(len(y), dtype=bool)
    for row, residual in residuals:
        if abs(residual) > cutoff:
            excluded[row] = True
    return excluded, cutoff


def jackknife_line(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    if len(x) < 4:
        slope, intercept = np.polyfit(x, y, 1)
        return float(slope), float(intercept)
    coefficients = []
    for held_out in range(len(x)):
        retained = np.arange(len(x)) != held_out
        coefficients.append(np.polyfit(x[retained], y[retained], 1))
    slope, intercept = np.median(np.asarray(coefficients), axis=0)
    return float(slope), float(intercept)


def fit_calibration(
    prediction: np.ndarray,
    y: np.ndarray,
    coal_types: np.ndarray,
    excluded: np.ndarray,
) -> dict[str, tuple[float, float]]:
    parameters: dict[str, tuple[float, float]] = {}
    for coal_type in np.unique(coal_types):
        mask = (coal_types == coal_type) & ~excluded
        if int(mask.sum()) < 3:
            mask = coal_types == coal_type
        parameters[str(coal_type)] = jackknife_line(prediction[mask], y[mask])
    return parameters


def apply_calibration(
    prediction: np.ndarray,
    coal_types: np.ndarray,
    parameters: dict[str, tuple[float, float]],
) -> np.ndarray:
    output = np.empty(len(prediction), dtype=float)
    for row, coal_type in enumerate(coal_types):
        slope, intercept = parameters[str(coal_type)]
        output[row] = slope * prediction[row] + intercept
    return output


def snv(x: np.ndarray) -> np.ndarray:
    values = np.asarray(x, float)
    center = values.mean(axis=1, keepdims=True)
    scale = values.std(axis=1, ddof=1, keepdims=True)
    return (values - center) / np.where(scale < 1.0e-12, 1.0, scale)


def transform_lvse(
    x_train: np.ndarray, x_test: np.ndarray, segments: int, modes: int
) -> tuple[np.ndarray, np.ndarray]:
    train_input = snv(x_train)
    test_input = snv(x_test)
    scaler = StandardScaler().fit(train_input)
    train_scaled = scaler.transform(train_input)
    test_scaled = scaler.transform(test_input)
    train_blocks = []
    test_blocks = []
    for indices in np.array_split(np.arange(x_train.shape[1]), segments):
        train_block = train_scaled[:, indices]
        test_block = test_scaled[:, indices]
        center = train_block.mean(axis=0)
        train_centered = train_block - center
        test_centered = test_block - center
        _, _, vectors = np.linalg.svd(train_centered, full_matrices=False)
        count = max(1, min(modes, vectors.shape[0], vectors.shape[1]))
        basis = vectors[:count].T
        train_blocks.append(train_centered @ basis)
        test_blocks.append(test_centered @ basis)
    return np.column_stack(train_blocks), np.column_stack(test_blocks)


def continuum_residual(
    spectra: np.ndarray, wavelengths: np.ndarray, band: tuple[float, ...]
) -> tuple[np.ndarray, np.ndarray]:
    low, high, left_low, left_high, right_low, right_high = band
    center_mask = (wavelengths >= low) & (wavelengths <= high)
    left_mask = (wavelengths >= left_low) & (wavelengths <= left_high)
    right_mask = (wavelengths >= right_low) & (wavelengths <= right_high)
    if min(center_mask.sum(), left_mask.sum(), right_mask.sum()) < 2:
        raise ValueError(f"Insufficient channels for emission band {low}-{high}")
    center_wavelengths = wavelengths[center_mask]
    left = spectra[:, left_mask].mean(axis=1)
    right = spectra[:, right_mask].mean(axis=1)
    left_center = 0.5 * (left_low + left_high)
    right_center = 0.5 * (right_low + right_high)
    fraction = (center_wavelengths - left_center) / (right_center - left_center)
    baseline = left[:, None] + (right - left)[:, None] * fraction[None, :]
    return center_wavelengths, spectra[:, center_mask] - baseline


def emission_bands(
    spectra: np.ndarray, wavelengths: np.ndarray, points: int = 64
) -> list[np.ndarray]:
    normalized = snv(spectra)
    target = np.linspace(0.0, 1.0, points)
    output = []
    for band in EMISSION_BANDS:
        local_wavelengths, residual = continuum_residual(normalized, wavelengths, band)
        local_axis = (local_wavelengths - local_wavelengths.min()) / (
            local_wavelengths.max() - local_wavelengths.min()
        )
        output.append(np.vstack([np.interp(target, local_axis, row) for row in residual]))
    return output


def pool_signed_rms(values: np.ndarray, bins: int = 4) -> np.ndarray:
    blocks = []
    for indices in np.array_split(np.arange(values.shape[1]), bins):
        part = values[:, indices]
        blocks.append(
            np.column_stack((part.mean(axis=1), np.sqrt(np.mean(part**2, axis=1))))
        )
    return np.column_stack(blocks)


def haar_features(band: np.ndarray, levels: int = 4) -> np.ndarray:
    scaling = np.asarray(band, float)
    output = []
    for level in range(1, levels + 1):
        shifted = np.roll(scaling, 2 ** (level - 1), axis=1)
        detail = 0.5 * (scaling - shifted)
        scaling = 0.5 * (scaling + shifted)
        output.append(pool_signed_rms(detail))
    output.append(
        np.column_stack(
            [scaling[:, indices].mean(axis=1) for indices in np.array_split(np.arange(scaling.shape[1]), 4)]
        )
    )
    return np.column_stack(output)


def geometry_scores(
    x_train_raw: np.ndarray, x_test_raw: np.ndarray, wavelengths: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    target = np.linspace(float(wavelengths.min()), float(wavelengths.max()), RAW_POINTS)
    train = np.vstack([np.interp(target, wavelengths, row) for row in x_train_raw]).astype(np.float32)
    test = np.vstack([np.interp(target, wavelengths, row) for row in x_test_raw]).astype(np.float32)
    coarse_train, coarse_test = transform_lvse(train, test, 16, 2)
    fine_train, fine_test = transform_lvse(train, test, 32, 4)
    train_wavelet = np.column_stack([haar_features(band) for band in emission_bands(train, target)])
    test_wavelet = np.column_stack([haar_features(band) for band in emission_bands(test, target)])
    train_features = np.column_stack((coarse_train, fine_train, train_wavelet)).astype(np.float32)
    test_features = np.column_stack((coarse_test, fine_test, test_wavelet)).astype(np.float32)
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_features)
    test_scaled = scaler.transform(test_features)
    components = min(10, len(train_features) - 1, train_features.shape[1])
    pca = PCA(n_components=components, svd_solver="full")
    train_scores = pca.fit_transform(train_scaled)
    test_scores = pca.transform(test_scaled)
    score_scale = np.maximum(np.std(train_scores, axis=0, ddof=1), 1.0e-12)
    return train_scores / score_scale, test_scores / score_scale


def train_final_model(
    project_root: Path,
    package_root: Path,
    prepared_path: Path,
    base_path: Path,
) -> tuple[Path, Path]:
    started = time.perf_counter()
    with np.load(prepared_path, allow_pickle=False) as data:
        prepared = {key: np.asarray(data[key]) for key in data.files}
    with np.load(base_path, allow_pickle=False) as base:
        raw_oof = np.asarray(base["raw_oof"], float)
        raw_test = np.asarray(base["raw_test"], float)
    y = prepared["y"].astype(float)
    coal_train = prepared["coal_train"].astype(str)
    coal_test = prepared["coal_test"].astype(str)
    excluded, cutoff = c_mad3_mask(raw_oof, y, coal_train)
    parameters = fit_calibration(raw_oof, y, coal_train, excluded)
    raw_train = apply_calibration(raw_oof, coal_train, parameters)
    raw_query = apply_calibration(raw_test, coal_test, parameters)
    train_scores, test_scores = geometry_scores(
        prepared["x_train"], prepared["x_test"], prepared["wavelengths"]
    )

    sample_path = project_root / "data" / "submit_sample" / "submit" / "submit.csv"
    sample = pd.read_csv(sample_path, encoding="utf-8-sig")
    sample_ids = sample.iloc[:, 0].astype(str).to_numpy()
    test_ids = prepared["test_ids"].astype(str)
    position = {batch_id: row for row, batch_id in enumerate(test_ids)}
    if set(position) != set(sample_ids):
        raise ValueError("Test batches do not match the submission template")
    order = np.asarray([position[batch_id] for batch_id in sample_ids], dtype=int)

    arrays: dict[str, np.ndarray] = {
        "batch_ids": np.asarray(sample_ids, dtype=str),
        "coal_train": np.asarray(coal_train, dtype=str),
        "coal_test": np.asarray(coal_test[order], dtype=str),
        "raw_test_calibrated": raw_query[order],
        "geometry_train_scores": train_scores,
        "geometry_test_scores": test_scores[order],
        "training_residuals": y - raw_train,
    }
    arrays["expected_prediction"] = predict_from_arrays(arrays)
    model_path = package_root / "model" / "model_weights.npz"
    np.savez_compressed(model_path, **arrays)
    submission_path = package_root / "submit.csv"
    pd.DataFrame(
        {
            sample.columns[0]: sample_ids,
            PREDICTION_COLUMN: arrays["expected_prediction"],
        }
    ).to_csv(
        submission_path,
        index=False,
        encoding="utf-8",
        float_format="%.10f",
        lineterminator="\n",
    )
    report = {
        "method": "fixed_multikernel_gpr_with_mean_preserving_residual_correction",
        "training_rows": int(len(y)),
        "test_rows": int(len(test_ids)),
        "fixed_pairs": [list(pair) for pair in FIXED_PAIRS],
        "pair_weights": PAIR_WEIGHTS.tolist(),
        "kernel_names": list(KERNEL_NAMES),
        "kernel_weights": KERNEL_WEIGHTS.tolist(),
        "group_folds": 5,
        "c_mad3_cutoff": float(cutoff),
        "excluded_training_rows": np.flatnonzero(excluded).astype(int).tolist(),
        "calibration_parameters": {key: list(value) for key, value in parameters.items()},
        "raw_oof_rmse": float(np.sqrt(np.mean((raw_oof - y) ** 2))),
        "calibrated_oof_rmse": float(np.sqrt(np.mean((raw_train - y) ** 2))),
        "prediction_min": float(arrays["expected_prediction"].min()),
        "prediction_max": float(arrays["expected_prediction"].max()),
        "prepared_data_sha256": file_sha256(prepared_path),
        "base_predictions_sha256": file_sha256(base_path),
        "weights_sha256": file_sha256(model_path),
        "submission_sha256": file_sha256(submission_path),
        "seconds": float(time.perf_counter() - started),
    }
    (package_root / "model" / "model_config.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"[final] oof_rmse={report['calibrated_oof_rmse']:.6f}, "
        f"excluded={report['excluded_training_rows']}, seconds={report['seconds']:.1f}",
        flush=True,
    )
    return model_path, submission_path


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True, encoding="utf-8")
    package_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--stage", choices=("data", "base", "final", "all"), default="all")
    parser.add_argument("--artifact-dir", type=Path, default=package_root / "training_artifacts")
    args = parser.parse_args()
    project_root = args.project_root.resolve()
    artifact_dir = args.artifact_dir.resolve()
    prepared_path = artifact_dir / "prepared_data.npz"
    base_path = package_root / "model" / "base_component_predictions.npz"

    if args.stage in {"data", "all"}:
        prepared_path = prepare_data(project_root, artifact_dir)
    if args.stage in {"base", "all"}:
        if not prepared_path.exists():
            raise FileNotFoundError("Run the data stage first")
        base_path = train_base_models(prepared_path, package_root / "model")
    if args.stage in {"final", "all"}:
        if not prepared_path.exists() or not base_path.exists():
            raise FileNotFoundError("Run the data and base stages first")
        train_final_model(project_root, package_root, prepared_path, base_path)


if __name__ == "__main__":
    main()
