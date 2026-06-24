"""Train readiness model with walk-forward CV and write eval/model_report.md."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

import joblib
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

from features.pipeline import CONTINUITY_DROP_PCT, load_features, load_gravityos_features
from models.baselines import NaiveAtTrend, NaiveGlobalMean, NaivePerExerciseMean

TARGET_COL = "performance_delta"
CATEGORICAL_COLS = ("exercise", "muscle_group", "split")
HUBER_ALPHA = 1.0

# Pre-workout features only — no same-day sets, volume, or intensity.
BASE_FEATURE_COLS = [
    "exercise",
    "muscle_group",
    "split",
    "volume_trailing_7d",
    "acwr",
    "days_since_last_session",
    "training_days_trailing_7d",
    "days_since_last_workout",
    "volume_wow_pct",
    "muscle_group_sessions_trailing_10d",
    "split_sessions_trailing_10d",
    "sleep_trailing_7d",
    "calories_trailing_7",
    "protein_trailing_7",
    "carbs_trailing_7",
    "resting_hr_trailing_7",
    "resting_hr_deviation",
    "bodyweight_kg",
    "bodyweight_trailing_7",
    "bodyweight_lag_1d",
    "sleep_deviation",
    "calories_deviation",
    "protein_deviation",
    "carbs_deviation",
    "sleep_lag_1d",
    "sleep_lag_2d",
    "resting_hr_lag_1d",
    "deload_flag",
    "day_of_week",
    "sleep_imputed",
    "calories_imputed",
    "protein_imputed",
    "carbs_imputed",
]

ARTIFACTS_DIR = Path("models/artifacts")
REPORT_PATH = Path("eval/model_report.md")
SHAP_PATH = Path("eval/shap_summary.png")
CALIBRATION_PATH = Path("eval/calibration_plot.png")
UNIVARIATE_PLOTS_PATH = Path("eval/feature_univariate_plots.png")
SYNTHETIC_META_PATH = Path("data/synthetic/meta.json")
UNIVARIATE_TOP_N = 8
UNIVARIATE_BINS = 10
UNIVARIATE_MIN_UNIQUE = 4
CALIBRATION_BINS = 10
CALIBRATION_AXIS_BUFFER_KG = 1.0

DEFAULT_N_FOLDS = 3
MIN_TRAIN_FRAC = 0.5


@dataclass
class FoldMetrics:
    fold: int
    n_train: int
    n_test: int
    test_start: str
    test_end: str
    lgb_mae: float
    linear_mae: float
    naive_mae: float
    global_mean_mae: float
    exercise_mean_mae: float


def add_target_delta(df: pd.DataFrame) -> pd.DataFrame:
    """
    Target: delta vs rolling mean e1RM for the **same exercise**.

    e1rm_trend uses the prior 3 sessions of this exercise within the same
    continuity segment (resets after equipment/gym discontinuity flags).
    """
    out = df.sort_values(["exercise", "session_date"]).copy()
    if "continuity_break" in out.columns:
        out["_continuity_segment"] = out.groupby("exercise")["continuity_break"].cumsum()
    else:
        out["_continuity_segment"] = 0
    out["e1rm_trend"] = out.groupby(["exercise", "_continuity_segment"])["top_set_e1rm_kg"].transform(
        lambda s: s.rolling(3, min_periods=1).mean().shift(1)
    )
    out["e1rm_trend"] = out["e1rm_trend"].fillna(out["top_set_e1rm_kg"])
    out[TARGET_COL] = out["top_set_e1rm_kg"] - out["e1rm_trend"]
    return out.drop(columns=["_continuity_segment"])


def training_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Exercise-sessions eligible for training/eval (excludes continuity breaks)."""
    if "continuity_break" not in df.columns:
        return df
    return df[df["continuity_break"] == 0].copy()


def resolve_feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in BASE_FEATURE_COLS if c in df.columns]


def _as_category(series: pd.Series) -> pd.Series:
    return series.fillna("unknown").astype(str).astype("category")


def prepare_features(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    X = df[feature_cols].copy()
    for col in CATEGORICAL_COLS:
        if col in X.columns:
            X[col] = _as_category(X[col])
    return X


def _categorical_indices(feature_cols: list[str]) -> list[int]:
    return [i for i, col in enumerate(feature_cols) if col in CATEGORICAL_COLS]


def _predict_lgbm(model: lgb.LGBMRegressor, X: pd.DataFrame) -> np.ndarray:
    return model.predict(X)


def _fit_lgbm(X_train: pd.DataFrame, y_train: pd.Series, feature_cols: list[str]) -> lgb.LGBMRegressor:
    model = lgb.LGBMRegressor(
        objective="huber",
        alpha=HUBER_ALPHA,
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        random_state=42,
        verbose=-1,
    )
    model.fit(
        X_train,
        y_train,
        categorical_feature=_categorical_indices(feature_cols),
    )
    return model


def _fit_linear(X_train: pd.DataFrame, y_train: pd.Series) -> tuple[LinearRegression, list[str]]:
    numeric = [c for c in X_train.columns if c not in CATEGORICAL_COLS]
    cat = [c for c in CATEGORICAL_COLS if c in X_train.columns]
    X_num = pd.get_dummies(X_train[cat + numeric], columns=cat, drop_first=False)
    model = LinearRegression()
    model.fit(X_num, y_train)
    return model, list(X_num.columns)


def _predict_linear(model, column_names: list[str], X: pd.DataFrame) -> np.ndarray:
    numeric = [c for c in X.columns if c not in CATEGORICAL_COLS]
    cat = [c for c in CATEGORICAL_COLS if c in X.columns]
    X_num = pd.get_dummies(X[cat + numeric], columns=cat, drop_first=False)
    X_num = X_num.reindex(columns=column_names, fill_value=0)
    return model.predict(X_num)


def walk_forward_splits(
    df: pd.DataFrame,
    n_folds: int = DEFAULT_N_FOLDS,
    min_train_frac: float = MIN_TRAIN_FRAC,
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Expanding-window walk-forward splits by calendar date.

    Each fold trains on all sessions before the test window and tests on the
    next chronological slice of dates. No future leakage into past.
    """
    df = df.sort_values("session_date").copy()
    dates = sorted(pd.to_datetime(df["session_date"].unique()))
    n_dates = len(dates)
    min_train_dates = max(1, int(n_dates * min_train_frac))
    remaining = n_dates - min_train_dates
    if remaining < 1:
        split_idx = int(len(df) * (1 - min_train_frac))
        return [(df.iloc[:split_idx], df.iloc[split_idx:])]

    # Split the remaining timeline into contiguous chunks and keep all dates.
    test_index_chunks = [
        chunk for chunk in np.array_split(np.arange(min_train_dates, n_dates), n_folds) if len(chunk) > 0
    ]
    folds: list[tuple[pd.DataFrame, pd.DataFrame]] = []

    for chunk in test_index_chunks:
        test_start_idx = int(chunk[0])
        test_end_idx = int(chunk[-1]) + 1
        train_dates = set(dates[:test_start_idx])
        test_dates = set(dates[test_start_idx:test_end_idx])
        train = df[df["session_date"].isin({d.date() for d in train_dates})]
        test = df[df["session_date"].isin({d.date() for d in test_dates})]
        if len(train) > 0 and len(test) > 0:
            folds.append((train, test))

    return folds if folds else [(df.iloc[: int(len(df) * min_train_frac)], df.iloc[int(len(df) * min_train_frac) :])]


def walk_forward_eval(
    df: pd.DataFrame,
    feature_cols: list[str],
    n_folds: int = DEFAULT_N_FOLDS,
) -> list[FoldMetrics]:
    naive = NaiveAtTrend()
    metrics: list[FoldMetrics] = []

    for i, (train_df, test_df) in enumerate(walk_forward_splits(df, n_folds=n_folds), start=1):
        X_train = prepare_features(train_df, feature_cols)
        X_test = prepare_features(test_df, feature_cols)
        y_train = train_df[TARGET_COL]
        y_test = test_df[TARGET_COL]

        lgbm = _fit_lgbm(X_train, y_train, feature_cols)
        linear, linear_cols = _fit_linear(X_train, y_train)
        global_mean = NaiveGlobalMean().fit(y_train)
        exercise_mean = NaivePerExerciseMean().fit(train_df, target_col=TARGET_COL)

        test_dates = pd.to_datetime(test_df["session_date"])
        metrics.append(
            FoldMetrics(
                fold=i,
                n_train=len(train_df),
                n_test=len(test_df),
                test_start=str(test_dates.min().date()),
                test_end=str(test_dates.max().date()),
                lgb_mae=mean_absolute_error(y_test, _predict_lgbm(lgbm, X_test)),
                linear_mae=mean_absolute_error(y_test, _predict_linear(linear, linear_cols, X_test)),
                naive_mae=mean_absolute_error(y_test, naive.predict(test_df)),
                global_mean_mae=mean_absolute_error(y_test, global_mean.predict(test_df)),
                exercise_mean_mae=mean_absolute_error(y_test, exercise_mean.predict(test_df)),
            )
        )

    return metrics


def walk_forward_predict(
    df: pd.DataFrame,
    feature_cols: list[str],
    n_folds: int = DEFAULT_N_FOLDS,
) -> pd.DataFrame:
    """Out-of-fold predictions from walk-forward CV (no leakage)."""
    naive = NaiveAtTrend()
    parts: list[pd.DataFrame] = []

    for fold, (train_df, test_df) in enumerate(walk_forward_splits(df, n_folds=n_folds), start=1):
        X_train = prepare_features(train_df, feature_cols)
        X_test = prepare_features(test_df, feature_cols)
        y_test = test_df[TARGET_COL]

        lgbm = _fit_lgbm(X_train, train_df[TARGET_COL], feature_cols)
        global_mean = NaiveGlobalMean().fit(train_df[TARGET_COL])
        exercise_mean = NaivePerExerciseMean().fit(train_df, target_col=TARGET_COL)
        parts.append(
            pd.DataFrame(
                {
                    "y_true": y_test.values,
                    "y_pred_lgb": _predict_lgbm(lgbm, X_test),
                    "y_pred_naive": naive.predict(test_df),
                    "y_pred_global_mean": global_mean.predict(test_df),
                    "y_pred_exercise_mean": exercise_mean.predict(test_df),
                    "fold": fold,
                },
                index=test_df.index,
            )
        )

    return pd.concat(parts)


def _binned_calibration(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    n_bins: int = 10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decile bins on predictions (equal count per bin when possible)."""
    try:
        bin_labels = pd.Series(pd.qcut(y_pred, q=n_bins, duplicates="drop"))
    except ValueError:
        return np.array([]), np.array([]), np.array([])

    centers: list[float] = []
    means: list[float] = []
    counts: list[int] = []

    for interval in bin_labels.cat.categories:
        mask = (bin_labels == interval).to_numpy()
        centers.append(float(y_pred[mask].mean()))
        means.append(float(y_true[mask].mean()))
        counts.append(int(mask.sum()))

    return np.array(centers), np.array(means), np.array(counts)


def _write_calibration_plot(
    oof: pd.DataFrame,
    out_path: Path,
    *,
    n_train: int,
) -> dict[str, float]:
    y = oof["y_true"].to_numpy()
    pred_lgb = oof["y_pred_lgb"].to_numpy()
    pred_naive = oof["y_pred_naive"].to_numpy()
    pred_global_mean = oof["y_pred_global_mean"].to_numpy()
    pred_exercise_mean = oof["y_pred_exercise_mean"].to_numpy()

    lgb_mae = float(mean_absolute_error(y, pred_lgb))
    naive_mae = float(mean_absolute_error(y, pred_naive))
    global_mean_mae = float(mean_absolute_error(y, pred_global_mean))
    exercise_mean_mae = float(mean_absolute_error(y, pred_exercise_mean))

    centers, means, counts = _binned_calibration(y, pred_lgb, n_bins=CALIBRATION_BINS)
    buf = CALIBRATION_AXIS_BUFFER_KG
    mean_actual = float(y.mean())
    n_oof = len(y)

    if len(centers):
        cal_lo = min(float(centers.min()), float(means.min()), 0.0, mean_actual) - buf
        cal_hi = max(float(centers.max()), float(means.max()), 0.0, mean_actual) + buf
    else:
        cal_lo, cal_hi = -buf, buf

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([cal_lo, cal_hi], [cal_lo, cal_hi], "k--", linewidth=1, label="Perfect calibration")
    if len(centers):
        sizes = 40 + 120 * (counts / counts.max())
        ax.scatter(centers, means, s=sizes, c="#2166ac", alpha=0.85, edgecolors="white", label="LightGBM (deciles)")
    ax.scatter(
        [0.0],
        [mean_actual],
        s=140,
        c="#ef8a62",
        marker="s",
        edgecolors="white",
        label=f"Naive (pred=0, mean actual={mean_actual:+.2f})",
        zorder=5,
    )
    ax.set_xlim(cal_lo, cal_hi)
    ax.set_ylim(cal_lo, cal_hi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Mean predicted performance_delta (kg)")
    ax.set_ylabel("Mean actual performance_delta (kg)")
    ax.set_title(
        f"OOF decile calibration (n≈{int(counts.mean()) if len(counts) else 0}/bin) — "
        f"LGB MAE {lgb_mae:.2f} vs naive {naive_mae:.2f} kg"
    )
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.axvline(0, color="gray", linewidth=0.5)

    fig.suptitle(
        f"Walk-forward OOF — {n_oof} sessions ({100 * n_oof / n_train:.0f}% of {n_train} training rows)",
        fontsize=10,
        y=1.02,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)

    return {
        "lgb_mae": lgb_mae,
        "naive_mae": naive_mae,
        "global_mean_mae": global_mean_mae,
        "exercise_mean_mae": exercise_mean_mae,
        "mean_actual": float(y.mean()),
        "n_oof": float(n_oof),
        "n_train": float(n_train),
        "pred_min": float(pred_lgb.min()),
        "pred_max": float(pred_lgb.max()),
    }


def _write_shap_plot(
    model: lgb.LGBMRegressor,
    X: pd.DataFrame,
    feature_cols: list[str],
    out_path: Path,
) -> list[tuple[str, float]]:
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X, show=False, max_display=min(20, len(feature_cols)))
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()

    mean_abs = np.abs(shap_values).mean(axis=0)
    return sorted(zip(feature_cols, mean_abs), key=lambda x: x[1], reverse=True)


def _decile_mean_curves(
    x: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    n_bins: int = 10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decile bins on feature x → (x center, mean actual, mean predicted)."""
    if len(x) < n_bins:
        return np.array([]), np.array([]), np.array([])
    try:
        labels = pd.Series(pd.qcut(x, q=n_bins, duplicates="drop"))
    except ValueError:
        return np.array([]), np.array([]), np.array([])

    centers: list[float] = []
    actual_means: list[float] = []
    pred_means: list[float] = []
    for interval in labels.cat.categories:
        mask = (labels == interval).to_numpy()
        centers.append(float(x[mask].mean()))
        actual_means.append(float(y_true[mask].mean()))
        pred_means.append(float(y_pred[mask].mean()))
    return np.array(centers), np.array(actual_means), np.array(pred_means)


def _is_univariate_plottable(merged: pd.DataFrame, feature: str) -> bool:
    """Decile curves need continuous-ish features — skip flags and near-constants."""
    if feature in CATEGORICAL_COLS or feature.endswith("_imputed"):
        return False
    if feature not in merged.columns:
        return False
    return int(merged[feature].nunique(dropna=True)) >= UNIVARIATE_MIN_UNIQUE


def _plot_numeric_univariate(ax: plt.Axes, merged: pd.DataFrame, feature: str) -> None:
    x = merged[feature].to_numpy(dtype=float)
    y_true = merged["y_true"].to_numpy()
    y_pred = merged["y_pred_lgb"].to_numpy()

    cx, cy_a, cy_p = _decile_mean_curves(x, y_true, y_pred, n_bins=UNIVARIATE_BINS)
    if len(cx):
        ax.plot(cx, cy_a, color="#2166ac", linewidth=2, marker="o", markersize=5, label="Actual Δ")
        ax.plot(
            cx,
            cy_p,
            color="#ef8a62",
            linewidth=2,
            linestyle="--",
            marker="s",
            markersize=4,
            label="Predicted Δ",
        )

    ax.axhline(0, color="gray", linewidth=0.5)
    ax.set_xlabel(feature)
    ax.set_ylabel("Mean performance_delta (kg)")
    ax.legend(loc="best", fontsize=6)
    ax.grid(True, alpha=0.25)


def _write_univariate_plots(
    df: pd.DataFrame,
    oof: pd.DataFrame,
    shap_ranked: list[tuple[str, float]],
    out_path: Path,
    *,
    top_n: int = UNIVARIATE_TOP_N,
) -> list[str]:
    """OOF binned line plots: feature vs mean actual/predicted performance_delta."""
    merged = df.join(oof[["y_true", "y_pred_lgb"]], how="inner")
    features: list[str] = []
    for name, _ in shap_ranked:
        if not _is_univariate_plottable(merged, name):
            continue
        features.append(name)
        if len(features) >= top_n:
            break
    if not features:
        return []

    ncols = 2
    nrows = (len(features) + 1) // 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 4.2 * nrows), squeeze=False)

    for ax, feat in zip(axes.flat, features):
        _plot_numeric_univariate(ax, merged, feat)
        shap_score = next((s for n, s in shap_ranked if n == feat), float("nan"))
        ax.set_title(f"{feat}  (mean |SHAP|={shap_score:.3f})", fontsize=9)

    for ax in axes.flat[len(features) :]:
        ax.axis("off")

    fig.suptitle(
        f"Univariate decile plots — OOF mean actual vs predicted Δ (n={len(merged)}, numeric only)",
        fontsize=11,
        y=1.01,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return features


def _synthetic_data_label(data_dir: Path) -> str:
    return f"Synthetic demo ({data_dir.as_posix()})"


def train(
    data_dir: Path | None = None,
    gravityos_dir: Path | None = None,
    n_folds: int = DEFAULT_N_FOLDS,
) -> None:
    if gravityos_dir is not None:
        df = load_gravityos_features(gravityos_dir)
        data_label = f"Gravity OS ({gravityos_dir})"
    else:
        assert data_dir is not None
        df = load_features(data_dir)
        data_label = _synthetic_data_label(data_dir)

    df = add_target_delta(df)
    n_sessions_raw = len(df)
    n_continuity_excluded = int(df["continuity_break"].sum()) if "continuity_break" in df.columns else 0
    df = training_rows(df)

    feature_cols = resolve_feature_cols(df)
    df = df.dropna(subset=feature_cols + [TARGET_COL])

    cv_metrics = walk_forward_eval(df, feature_cols, n_folds=n_folds)
    oof = walk_forward_predict(df, feature_cols, n_folds=n_folds)
    cal_stats = _write_calibration_plot(oof, CALIBRATION_PATH, n_train=len(df))

    # Final deployment model: all data, most recent volume feature choice.
    X_all = prepare_features(df, feature_cols)
    y_all = df[TARGET_COL]
    lgbm = _fit_lgbm(X_all, y_all, feature_cols)

    # SHAP on the last fold's test set (held-out recent window).
    last_train, last_test = walk_forward_splits(df, n_folds=n_folds)[-1]
    X_shap = prepare_features(last_test, feature_cols)
    shap_ranked = _write_shap_plot(lgbm, X_shap, feature_cols, SHAP_PATH)
    univariate_features = _write_univariate_plots(df, oof, shap_ranked, UNIVARIATE_PLOTS_PATH)

    last_fold = cv_metrics[-1]
    linear, linear_cols = _fit_linear(prepare_features(last_train, feature_cols), last_train[TARGET_COL])
    linear_mae = mean_absolute_error(
        last_test[TARGET_COL],
        _predict_linear(linear, linear_cols, prepare_features(last_test, feature_cols)),
    )
    naive_mae = mean_absolute_error(last_test[TARGET_COL], NaiveAtTrend().predict(last_test))
    global_mean_mae = mean_absolute_error(
        last_test[TARGET_COL],
        NaiveGlobalMean().fit(last_train[TARGET_COL]).predict(last_test),
    )
    exercise_mean_mae = mean_absolute_error(
        last_test[TARGET_COL],
        NaivePerExerciseMean().fit(last_train, target_col=TARGET_COL).predict(last_test),
    )

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(lgbm, ARTIFACTS_DIR / "lgb_readiness.pkl")
    joblib.dump(
        {
            "feature_cols": feature_cols,
            "target_col": TARGET_COL,
            "categorical_cols": list(CATEGORICAL_COLS),
            "objective": "huber",
            "huber_alpha": HUBER_ALPHA,
            "scoring_mode": "pre_workout",
            "continuity_drop_pct": CONTINUITY_DROP_PCT,
        },
        ARTIFACTS_DIR / "model_meta.pkl",
    )

    report = _build_report(
        data_label=data_label,
        cv_metrics=cv_metrics,
        last_fold_lgb_mae=last_fold.lgb_mae,
        last_fold_linear_mae=linear_mae,
        last_fold_naive_mae=naive_mae,
        last_fold_global_mean_mae=global_mean_mae,
        last_fold_exercise_mean_mae=exercise_mean_mae,
        n_total=len(df),
        n_sessions_raw=n_sessions_raw,
        n_continuity_excluded=n_continuity_excluded,
        shap_ranked=shap_ranked,
        univariate_features=univariate_features,
        n_folds=n_folds,
        cal_stats=cal_stats,
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(f"Wrote {REPORT_PATH}")
    print(f"Wrote {SHAP_PATH}")
    print(f"Wrote {CALIBRATION_PATH}")
    print(f"Wrote {UNIVARIATE_PLOTS_PATH}")
    print(f"Continuity exclusions: {n_continuity_excluded} exercise-sessions (>{CONTINUITY_DROP_PCT:.0%} weight drop)")
    print(f"Training rows: {len(df)} (from {n_sessions_raw} raw)")
    print(
        f"Walk-forward LightGBM MAE: "
        f"{np.mean([m.lgb_mae for m in cv_metrics]):.2f} ± {np.std([m.lgb_mae for m in cv_metrics]):.2f}"
    )


def _build_report(
    data_label: str,
    cv_metrics: list[FoldMetrics],
    last_fold_lgb_mae: float,
    last_fold_linear_mae: float,
    last_fold_naive_mae: float,
    last_fold_global_mean_mae: float,
    last_fold_exercise_mean_mae: float,
    n_total: int,
    n_sessions_raw: int,
    n_continuity_excluded: int,
    shap_ranked: list[tuple[str, float]],
    univariate_features: list[str],
    n_folds: int,
    cal_stats: dict[str, float],
) -> str:
    shap_lines = "\n".join(
        f"| {name} | {score:.3f} |" for name, score in shap_ranked[:12]
    )
    fold_lines = "\n".join(
        f"| {m.fold} | {m.test_start} → {m.test_end} | {m.n_train} | {m.n_test} | "
        f"{m.lgb_mae:.2f} | {m.linear_mae:.2f} | {m.naive_mae:.2f} | {m.global_mean_mae:.2f} | {m.exercise_mean_mae:.2f} |"
        for m in cv_metrics
    )
    lgb_mean = np.mean([m.lgb_mae for m in cv_metrics])
    lgb_std = np.std([m.lgb_mae for m in cv_metrics])
    naive_mean = np.mean([m.naive_mae for m in cv_metrics])
    global_mean = np.mean([m.global_mean_mae for m in cv_metrics])
    exercise_mean = np.mean([m.exercise_mean_mae for m in cv_metrics])
    is_synthetic = "synthetic" in data_label.lower()
    if is_synthetic:
        data_caveat = (
            "- **Synthetic demo data** — `data/synthetic`. No personal workout/recovery rows are in this repo.\n"
            "- Reproduce these metrics: `python -m models.train --data-dir data/synthetic`.\n"
        )
    else:
        data_caveat = "- Trained on personal logs — do not commit artifacts or reports with identifiable data.\n"
    return f"""# Model Report

> Auto-generated by `python -m models.train`. Re-run after feature or data changes.

## Validation

- **Data source:** {data_label}
- **Split:** expanding-window walk-forward ({n_folds} folds, min {MIN_TRAIN_FRAC:.0%} of timeline for first train window)
- **Target:** `performance_delta` (kg) — top-set e1RM minus prior-3-**same-exercise**-session rolling mean (not vs last session alone)
- **Model:** LightGBM (Huber loss, α={HUBER_ALPHA}) — **pre-workout features only** (no same-day sets/volume/intensity); categorical `exercise`, `muscle_group`, `split`
- **Evaluation policy:** MAE and calibration use raw model predictions (no post-processing).
- **Continuity filter:** exclude exercise-sessions with max working weight drop > {CONTINUITY_DROP_PCT:.0%} vs prior same-exercise log ({n_continuity_excluded} excluded, {n_sessions_raw} raw rows)
- **Training rows:** {n_total} (after continuity filter + feature completeness)

## Walk-forward CV — MAE on performance_delta (kg)

| Fold | Test period | Train n | Test n | LightGBM | Linear | Naive (0) | Naive (global mean) | Naive (per-exercise mean) |
|------|-------------|---------|--------|----------|--------|-----------|----------------------|---------------------------|
{fold_lines}

**LightGBM mean ± std:** {lgb_mean:.2f} ± {lgb_std:.2f} kg  
**Naive mean (0):** {naive_mean:.2f} kg  
**Naive mean (global):** {global_mean:.2f} kg  
**Naive mean (per-exercise):** {exercise_mean:.2f} kg

## Last fold (most recent held-out window)

| Model | MAE |
|-------|-----|
| LightGBM | {last_fold_lgb_mae:.2f} |
| Linear baseline | {last_fold_linear_mae:.2f} |
| Naive (at trend = 0) | {last_fold_naive_mae:.2f} |
| Naive (global train mean) | {last_fold_global_mean_mae:.2f} |
| Naive (per-exercise train mean) | {last_fold_exercise_mean_mae:.2f} |

## Calibration (walk-forward OOF)

![Calibration plot](calibration_plot.png)

- **OOF sessions:** {cal_stats['n_oof']:.0f} of {cal_stats['n_train']:.0f} training rows ({100 * cal_stats['n_oof'] / cal_stats['n_train']:.0f}%; first {MIN_TRAIN_FRAC:.0%} of timeline is train-only burn-in)
- **OOF pred range:** {cal_stats['pred_min']:+.2f} to {cal_stats['pred_max']:+.2f} kg
- **OOF LightGBM MAE:** {cal_stats['lgb_mae']:.2f} kg
- **OOF Naive MAE:** {cal_stats['naive_mae']:.2f} kg
- **OOF Naive (global mean) MAE:** {cal_stats['global_mean_mae']:.2f} kg
- **OOF Naive (per-exercise mean) MAE:** {cal_stats['exercise_mean_mae']:.2f} kg
- **Mean actual delta:** {cal_stats['mean_actual']:+.2f} kg (naive bias if predicting 0)

Decile bins on OOF predictions (equal count per bin). Axes fit data ± {CALIBRATION_AXIS_BUFFER_KG:.0f} kg. Points on the diagonal = well calibrated.

## SHAP — mean |contribution| (last fold test set)

![SHAP summary](shap_summary.png)

| Feature | mean \\|SHAP\\| |
|---------|-------------|
{shap_lines}

## Univariate feature plots (OOF)

![Feature univariate plots](feature_univariate_plots.png)

Top-{len(univariate_features)} numeric SHAP features: {", ".join(f"`{f}`" for f in univariate_features)}. Feature split into deciles (x = mean feature value per decile); y = mean actual (blue) vs mean OOF predicted (orange) `performance_delta`.

## Limitations

{data_caveat}- Exercise-sessions flagged `continuity_break` (likely equipment/gym change) are excluded from train/eval; e1RM trend resets per segment.
- Correlation ≠ causation for sleep/calorie/macro features.
- Walk-forward LightGBM modestly beats naive-at-trend on this demo set; last fold can be near parity.
- Random splits would inflate metrics — walk-forward only.
- Final artifact is fit on **all** data after CV evaluation.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Train readiness model")
    parser.add_argument("--data-dir", type=Path, default=Path("data/synthetic"))
    parser.add_argument("--gravityos", action="store_true")
    parser.add_argument("--gravityos-dir", type=Path, default=None)
    parser.add_argument("--folds", type=int, default=DEFAULT_N_FOLDS)
    args = parser.parse_args()

    if args.gravityos:
        gravityos_dir = args.gravityos_dir or Path(os.environ["GRAVITYOS_DATA_DIR"])
        train(gravityos_dir=gravityos_dir, n_folds=args.folds)
    else:
        train(data_dir=args.data_dir, n_folds=args.folds)


if __name__ == "__main__":
    main()
