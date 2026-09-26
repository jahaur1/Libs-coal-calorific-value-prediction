"""Reusable data loading, spectroscopy, GPR, and calibration primitives."""
from __future__ import annotations

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

warnings.filterwarnings("ignore")


# 光谱预处理、VIP/GPR与残差几何配置。
SEED = 0
FFT_CUTOFF = 0.02
VIP_THRESHOLD = 1.0
VIP_COMPONENTS = 20
GPR_RESTARTS = 2
MASK_AGGREGATES = 5
RAW_POINTS = 512
KERNEL_NAMES = ("rbf", "matern15", "matern25", "rational_quadratic", "dot_product")
# 无标题标签表的列映射。
LABEL_COLUMNS = (
    "batch_day",
    "total_moisture",
    "analysis_moisture",
    "ash",
    "target_q",
    "hydrogen",
    "sulfur",
)
# 四个发射波段，每项依次为中心区间、左连续谱区间和右连续谱区间。
EMISSION_BANDS = (
    (306.0, 310.0, 302.0, 305.0, 311.0, 314.0),
    (484.0, 488.0, 480.0, 483.0, 489.0, 492.0),
    (653.0, 660.0, 648.0, 652.0, 661.0, 665.0),
    (774.0, 781.0, 768.0, 773.0, 782.0, 788.0),
)


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


def resolve_competition_data_root(xfdata_root: Path) -> Path:
    """Locate the organizer-provided dataset without relying on machine paths."""
    root = xfdata_root.resolve()
    required = ("训练集", "训练集标签", "测试集")
    if all((root / name).is_dir() for name in required):
        return root
    candidates = sorted(
        {
            path.parent
            for path in root.rglob("训练集")
            if path.is_dir()
            and all((path.parent / name).is_dir() for name in required)
        }
    )
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"Expected one dataset containing {required} under {root}; found {len(candidates)}"
        )
    return candidates[0]


def submission_template_path(xfdata_root: Path) -> Path:
    data_root = resolve_competition_data_root(xfdata_root)
    candidates = sorted(data_root.glob("submit_sample/**/submit.csv"))
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"Expected one submit_sample/**/submit.csv under {data_root}; found {len(candidates)}"
        )
    return candidates[0]


def build_batch_index(xfdata_root: Path) -> pd.DataFrame:
    data_root = resolve_competition_data_root(xfdata_root)
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
    # 竞赛光谱CSV前4行为仪器元数据，后续前两列分别是波长和强度。
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


def load_competition_arrays(xfdata_root: Path) -> dict[str, np.ndarray]:
    """Read the organizer data and return deterministic batch-mean arrays."""
    started = time.perf_counter()
    index = build_batch_index(xfdata_root)
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
    arrays = {
        "wavelengths": reference_wavelengths,
        "x_train": spectra[train_mask],
        "x_test": spectra[test_mask],
        "y": index.loc[train_mask, "target_q"].to_numpy(np.float32),
        "train_ids": np.asarray(index.loc[train_mask, "batch_name"].astype(str), dtype=str),
        "test_ids": np.asarray(index.loc[test_mask, "batch_name"].astype(str), dtype=str),
        "coal_train": np.asarray(index.loc[train_mask, "coal_type"].astype(str), dtype=str),
        "coal_test": np.asarray(index.loc[test_mask, "coal_type"].astype(str), dtype=str),
    }
    print(
        f"[data] train={int(train_mask.sum())}, test={int(test_mask.sum())}, "
        f"channels={spectra.shape[1]}, seconds={time.perf_counter() - started:.1f}",
        flush=True,
    )
    return arrays


def prepare_data(xfdata_root: Path, artifact_dir: Path) -> Path:
    arrays = load_competition_arrays(xfdata_root)
    path = artifact_dir / "prepared_data.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        **arrays,
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
    # 二阶Savitzky-Golay多项式用于估计一阶导数；窗口长度来自ALL_PAIRS候选。
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
    # 统一的核参数初始值。
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
    # 固定5折生成基础模型OOF；每个批次均值只占一行，批次不会跨折。
    splitter = GroupKFold(5)
    for train_rows, valid_rows in splitter.split(x, y, groups):
        np.random.seed(SEED)
        mask = select_vip_mask(x[train_rows], y[train_rows])
        # 少于5个VIP通道时回退为全通道，避免GPR特征空间退化。
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
    prediction, _, _ = fit_test_kernel_with_state(
        x_train, y, x_test, kernel_name, radius
    )
    return prediction


def full_training_masks(x_train: np.ndarray, y: np.ndarray) -> list[np.ndarray]:
    """Create the deterministic VIP mask ensemble used by the frozen model."""
    np.random.seed(SEED)
    base_mask = select_vip_mask(x_train, y)
    if int(base_mask.sum()) < 5:
        base_mask = np.ones(x_train.shape[1], dtype=bool)
    masks = [base_mask]
    for aggregate in range(1, MASK_AGGREGATES):
        generator = np.random.RandomState(SEED + 1000 + aggregate)
        mask = base_mask.copy()
        selected = np.flatnonzero(mask)
        mask[selected[generator.rand(len(selected)) < 0.05]] = False
        available = np.flatnonzero(~mask)
        extra_count = min(int(0.02 * len(mask)), len(available))
        if extra_count:
            mask[generator.choice(available, size=extra_count, replace=False)] = True
        masks.append(mask)
    return masks


def fit_test_kernel_with_state(
    x_train: np.ndarray,
    y: np.ndarray,
    x_test: np.ndarray,
    kernel_name: str,
    radius: int,
) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
    """Fit one full-data branch and return its optimized kernel parameters."""
    masks = full_training_masks(x_train, y)
    predictions: list[np.ndarray] = []
    kernel_thetas: list[np.ndarray] = []
    for mask in masks:
        train_features = augment(x_train, mask, radius)
        test_features = augment(x_test, mask, radius)
        scaler = StandardScaler()
        train_scaled = scaler.fit_transform(train_features)
        test_scaled = scaler.transform(test_features)
        np.random.seed(SEED)
        model = make_gpr(kernel_name)
        model.fit(train_scaled, y)
        predictions.append(model.predict(test_scaled))
        kernel_thetas.append(np.asarray(model.kernel_.theta, dtype=np.float64))
    return np.mean(predictions, axis=0), masks, kernel_thetas


def predict_test_kernel_from_state(
    x_train: np.ndarray,
    y: np.ndarray,
    x_test: np.ndarray,
    kernel_name: str,
    radius: int,
    masks: list[np.ndarray],
    kernel_thetas: list[np.ndarray],
) -> np.ndarray:
    """Reconstruct a fitted GPR branch without optimizing any hyperparameter."""
    if len(masks) != MASK_AGGREGATES or len(kernel_thetas) != MASK_AGGREGATES:
        raise ValueError("Frozen GPR state has an unexpected aggregate count")
    predictions: list[np.ndarray] = []
    for mask, theta in zip(masks, kernel_thetas):
        train_features = augment(x_train, np.asarray(mask, bool), radius)
        test_features = augment(x_test, np.asarray(mask, bool), radius)
        scaler = StandardScaler()
        train_scaled = scaler.fit_transform(train_features)
        test_scaled = scaler.transform(test_features)
        fitted_kernel = make_gpr(kernel_name).kernel.clone_with_theta(
            np.asarray(theta, dtype=float)
        )
        model = GaussianProcessRegressor(
            kernel=fitted_kernel,
            optimizer=None,
            normalize_y=True,
            random_state=SEED,
        )
        model.fit(train_scaled, y)
        predictions.append(model.predict(test_scaled))
    return np.mean(predictions, axis=0)


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
    # C_mad3定义：绝对校准残差中位数加3倍稳健MAD作为训练异常阈值。
    cutoff = median + 3.0 * scaled_mad
    excluded = np.zeros(len(y), dtype=bool)
    for row, residual in residuals:
        if abs(residual) > cutoff:
            excluded[row] = True
    return excluded, cutoff


def jackknife_line(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    # 少于4个样本无法稳定执行留一法，因此回退为普通一次线性拟合。
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
        # 剔除异常后少于3个样本时恢复使用该煤种全部训练样本，避免无法拟合直线。
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
    # 每个局部波段统一插值为64点，保证不同原始波长网格下特征维度一致。
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
    # 每个Haar尺度固定汇总为4个局部区块，并保留有符号均值与RMS能量。
    blocks = []
    for indices in np.array_split(np.arange(values.shape[1]), bins):
        part = values[:, indices]
        blocks.append(
            np.column_stack((part.mean(axis=1), np.sqrt(np.mean(part**2, axis=1))))
        )
    return np.column_stack(blocks)


def haar_features(band: np.ndarray, levels: int = 4) -> np.ndarray:
    # 使用4个Haar尺度覆盖由局部到较宽的波段变化。
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
    # 两个预先声明的LVSE视图：粗视图16段×2模态，细视图32段×4模态。
    coarse_train, coarse_test = transform_lvse(train, test, 16, 2)
    fine_train, fine_test = transform_lvse(train, test, 32, 4)
    train_wavelet = np.column_stack([haar_features(band) for band in emission_bands(train, target)])
    test_wavelet = np.column_stack([haar_features(band) for band in emission_bands(test, target)])
    train_features = np.column_stack((coarse_train, fine_train, train_wavelet)).astype(np.float32)
    test_features = np.column_stack((coarse_test, fine_test, test_wavelet)).astype(np.float32)
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_features)
    test_scaled = scaler.transform(test_features)
    # 几何坐标最多保留10维。
    components = min(10, len(train_features) - 1, train_features.shape[1])
    pca = PCA(n_components=components, svd_solver="full")
    train_scores = pca.fit_transform(train_scaled)
    test_scores = pca.transform(test_scaled)
    score_scale = np.maximum(np.std(train_scores, axis=0, ddof=1), 1.0e-12)
    return train_scores / score_scale, test_scores / score_scale
