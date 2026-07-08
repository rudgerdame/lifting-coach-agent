"""Sweep synthetic-generator coefficients to find a config with target classifier AUC.

Regenerates synthetic data for each (config, seed), runs the walk-forward readiness
classifier, and reports macro / per-class OOF ROC AUC. Used to tune the demo data so
the readiness classifier shows a clear (~0.70) signal.

Usage:
    python -m scripts.sweep_synthetic
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import ingestion.synthetic as syn
from features.pipeline import load_features
from models.train import (
    DEFAULT_K,
    _clf_oof_stats,
    add_target_delta,
    resolve_feature_cols,
    training_rows,
    walk_forward_classify,
)

# Each config overrides a subset of the generator's module-level coefficients.
# Baseline (current file values) is included as "current".
CONFIGS: dict[str, dict[str, float]] = {
    "current": {},
    "A_strong_lag1": {
        "_READINESS_AR": 0.70,
        "_READINESS_SHOCK_STD": 0.40,
        "_SLEEP_LAG1_COEF": 2.3,
        "_SLEEP_LAG2_COEF": 1.0,
        "_RHR_TRAIL7_COEF": -0.12,
        "_RESIDUAL_NOISE_STD": 0.10,
        "_REP_READINESS_COEF": 1.0,
        "_REP_PER_SET_NOISE": 0.35,
    },
    "B_lowAR_lownoise": {
        "_READINESS_AR": 0.55,
        "_READINESS_SHOCK_STD": 0.50,
        "_SLEEP_LAG1_COEF": 2.6,
        "_SLEEP_LAG2_COEF": 1.1,
        "_RHR_TRAIL7_COEF": -0.14,
        "_RESIDUAL_NOISE_STD": 0.08,
        "_REP_READINESS_COEF": 1.2,
        "_REP_PER_SET_NOISE": 0.30,
    },
    "C_highAR_modsignal": {
        "_READINESS_AR": 0.85,
        "_READINESS_SHOCK_STD": 0.35,
        "_SLEEP_LAG1_COEF": 2.0,
        "_SLEEP_LAG2_COEF": 0.9,
        "_RHR_TRAIL7_COEF": -0.11,
        "_RESIDUAL_NOISE_STD": 0.10,
        "_REP_READINESS_COEF": 1.0,
        "_REP_PER_SET_NOISE": 0.40,
    },
    "D_repdominant": {
        "_READINESS_AR": 0.60,
        "_READINESS_SHOCK_STD": 0.45,
        "_SLEEP_LAG1_COEF": 2.2,
        "_SLEEP_LAG2_COEF": 1.0,
        "_RHR_TRAIL7_COEF": -0.13,
        "_RESIDUAL_NOISE_STD": 0.08,
        "_REP_READINESS_COEF": 1.5,
        "_REP_PER_SET_NOISE": 0.30,
    },
    "E_verylownoise": {
        "_READINESS_AR": 0.65,
        "_READINESS_SHOCK_STD": 0.45,
        "_SLEEP_LAG1_COEF": 2.4,
        "_SLEEP_LAG2_COEF": 1.1,
        "_RHR_TRAIL7_COEF": -0.14,
        "_VOLUME_TRAIL7_COEF": -0.00055,
        "_TRAINING_DAYS_COEF": -0.55,
        "_RESIDUAL_NOISE_STD": 0.06,
        "_REP_READINESS_COEF": 1.3,
        "_REP_PER_SET_NOISE": 0.28,
    },
}

SEEDS = (42, 7, 123)


def _apply(overrides: dict[str, float]) -> dict[str, float]:
    saved = {name: getattr(syn, name) for name in overrides}
    for name, value in overrides.items():
        setattr(syn, name, value)
    return saved


def _restore(saved: dict[str, float]) -> None:
    for name, value in saved.items():
        setattr(syn, name, value)


def _evaluate(seed: int) -> dict[str, float]:
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        syn.write_synthetic(out, seed=seed)
        df = load_features(out)
    df = add_target_delta(df)
    df = training_rows(df)
    feature_cols = resolve_feature_cols(df)
    df = df.dropna(subset=feature_cols + ["performance_delta"])
    _, oof = walk_forward_classify(df, feature_cols, k=DEFAULT_K)
    stats = _clf_oof_stats(oof)
    auc = stats["per_class_auc"]
    return {
        "n": stats["n"],
        "macro_auc": stats["macro_auc"],
        "below": float(auc[0]),
        "at": float(auc[1]),
        "above": float(auc[2]),
        "accuracy": stats["accuracy"],
    }


def scan_seeds(seeds: tuple[int, ...]) -> None:
    """Evaluate the *current* generator constants across seeds (no overrides)."""
    for seed in seeds:
        res = _evaluate(seed)
        print(
            f"seed={seed:<5d} n={res['n']:<5d} macroAUC={res['macro_auc']:.3f} "
            f"acc={res['accuracy']:.3f}  below={res['below']:.3f} at={res['at']:.3f} above={res['above']:.3f}"
        )


def main() -> None:
    import sys

    if "--scan-seeds" in sys.argv:
        scan_seeds((42, 7, 123, 5, 11, 17, 21, 99, 202, 2024))
        return

    rows: list[tuple[str, int, dict[str, float]]] = []
    for name, overrides in CONFIGS.items():
        saved = _apply(overrides)
        try:
            for seed in SEEDS:
                res = _evaluate(seed)
                rows.append((name, seed, res))
                print(
                    f"{name:22s} seed={seed:<4d} n={res['n']:<5d} "
                    f"macroAUC={res['macro_auc']:.3f}  "
                    f"below={res['below']:.3f} at={res['at']:.3f} above={res['above']:.3f}  "
                    f"acc={res['accuracy']:.3f}"
                )
        finally:
            _restore(saved)

    print("\n=== Mean macro AUC across seeds ===")
    by_config: dict[str, list[float]] = {}
    for name, _seed, res in rows:
        by_config.setdefault(name, []).append(res["macro_auc"])
    ranked = sorted(by_config.items(), key=lambda kv: sum(kv[1]) / len(kv[1]), reverse=True)
    for name, aucs in ranked:
        mean = sum(aucs) / len(aucs)
        print(f"{name:22s} mean_macro_auc={mean:.3f}  (runs: {', '.join(f'{a:.3f}' for a in aucs)})")
    best = ranked[0][0]
    print(f"\nBest config: {best}  -> apply its overrides to ingestion/synthetic.py")


if __name__ == "__main__":
    main()
