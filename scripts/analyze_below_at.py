"""Analyze actual-below / predicted-at misclassifications."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from features.pipeline import load_features
from models.labeling import DeltaClassLabeler
from models.train import (
    DEFAULT_K,
    add_target_delta,
    resolve_feature_cols,
    training_rows,
    walk_forward_classify,
)


def main() -> None:
    df = load_features(Path("data/synthetic"))
    df = add_target_delta(df)
    df = training_rows(df)
    fc = resolve_feature_cols(df)
    df = df.dropna(subset=fc + ["performance_delta"])

    _, oof = walk_forward_classify(df, fc, k=DEFAULT_K)
    merged = df.join(oof, how="inner")

    actual_below = merged["y_true"] == 0
    pred_at = merged["y_pred"] == 1
    mask = actual_below & pred_at
    n_miss = int(mask.sum())
    n_below = int(actual_below.sum())
    print(f"Actual below_trend -> predicted at_trend: {n_miss} / {n_below} ({100 * n_miss / n_below:.0f}%)")
    print(f"Actual below -> predicted above: {int((actual_below & (merged['y_pred'] == 2)).sum())}")
    print(f"Actual below -> correct: {int((actual_below & (merged['y_pred'] == 0)).sum())}")

    sub = merged[mask]
    ok = merged[actual_below & (merged["y_pred"] == 0)]

    print("\n--- Misclassified below->at: calibrated probs ---")
    print(sub[["p_below", "p_at", "p_above"]].describe().round(3).to_string())

    print("\n--- Misclassified below->at: actual performance_delta (kg) ---")
    print(sub["performance_delta"].describe().round(2).to_string())

    if len(ok):
        print("\n--- Correctly classified below: actual performance_delta (kg) ---")
        print(ok["performance_delta"].describe().round(2).to_string())
        print("\n--- Correctly classified below: calibrated probs ---")
        print(ok[["p_below", "p_at", "p_above"]].describe().round(3).to_string())

    lab = DeltaClassLabeler(k=DEFAULT_K).fit(df)
    thr = lab._thresholds(sub)
    delta = sub["performance_delta"].to_numpy(dtype=float)
    # How far past the below boundary (0 = exactly at boundary, 1 = one threshold width below)
    margin = (-delta - thr) / np.maximum(thr, 1e-6)
    print("\n--- Misclassified below->at: depth below boundary (in threshold widths) ---")
    print(f"mean={margin.mean():.2f}  median={np.median(margin):.2f}  max={margin.max():.2f}")
    print(f"within 0.5 threshold widths of at/below boundary: {(margin < 0.5).mean():.0%}")

    for feat in ["sleep_lag_1d", "sleep_deviation", "resting_hr_lag_1d", "acwr", "deload_flag"]:
        if feat not in sub.columns:
            continue
        at_actual = merged[merged["y_true"] == 1]
        print(
            f"\n{feat}: miss->at mean={sub[feat].mean():.2f}  "
            f"correct-below mean={ok[feat].mean():.2f}  "
            f"actual-at mean={at_actual[feat].mean():.2f}"
        )


if __name__ == "__main__":
    main()
