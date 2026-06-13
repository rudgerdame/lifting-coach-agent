"""Recovery anomaly detection — hybrid rule + personalized statistical flags."""

from __future__ import annotations

import numpy as np
import pandas as pd

# Hard floors: values below these are almost always logging/sync failures.
SLEEP_MIN_HOURS = 4.0
CALORIES_MIN_KCAL = 1000.0
PROTEIN_MIN_G = 30.0
CARBS_MIN_G = 50.0

# Personalized lower-tail detection on prior valid history.
ANOMALY_MAD_K = 3.0
ANOMALY_BASELINE_DAYS = 28
ANOMALY_MIN_HISTORY = 7
MAD_SCALE = 1.4826  # scale MAD to approximate std under normality


def hard_floor_anomalies(series: pd.Series, floor: float) -> pd.Series:
    """Flag values below an absolute floor (instrumentation failure)."""
    return series < floor


def mad_lower_tail_anomalies(
    series: pd.Series,
    *,
    exclude: pd.Series | None = None,
    k: float = ANOMALY_MAD_K,
    window: int = ANOMALY_BASELINE_DAYS,
    min_history: int = ANOMALY_MIN_HISTORY,
) -> pd.Series:
    """
    Flag lower-tail outliers vs a trailing personal baseline.

    For each day, compare the value to the median of the prior `window` valid
    days. Flag if value < median - k * scaled_MAD. Only the lower tail is
    tested — we care about missing syncs, not unusually high readings.
    """
    exclude = pd.Series(False, index=series.index) if exclude is None else exclude
    anomalies = pd.Series(False, index=series.index)
    trusted = series.where(~exclude)

    for i in range(len(series)):
        if exclude.iloc[i]:
            continue
        history = trusted.iloc[max(0, i - window) : i].dropna()
        if len(history) < min_history:
            continue
        med = history.median()
        mad = (history - med).abs().median()
        if mad == 0:
            continue
        threshold = med - k * MAD_SCALE * mad
        if series.iloc[i] < threshold:
            anomalies.iloc[i] = True

    return anomalies


def impute_with_trailing_valid_mean(
    series: pd.Series,
    anomaly_mask: pd.Series,
    *,
    window: int = 7,
    fallback: float | None = None,
) -> pd.Series:
    """Fill anomalous rows from the mean of valid readings in the prior `window` days."""
    valid = series.where(~anomaly_mask)
    trailing = valid.rolling(window, min_periods=1).mean().shift(1)
    imputed = series.where(~anomaly_mask, trailing)
    if fallback is not None and pd.notna(fallback):
        imputed = imputed.fillna(fallback)
    return imputed
