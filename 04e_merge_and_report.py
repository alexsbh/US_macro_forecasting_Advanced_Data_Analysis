"""04e_merge_and_report.py

Purpose
- Combine forecasts produced by the different modeling scripts into one coherent
  evaluation dataset.
- Recompute metrics and pairwise accuracy comparisons in a single place so that
  results are consistent across models.

What the script does
1) Loads per-horizon forecast files from:
   - baseline models (VAR / ridge),
   - XGBoost,
   - RNN models (LSTM / GRU).
2) Merges forecasts on the common time index and target definition so that every
   model is evaluated on the same realized values.
3) Computes standard accuracy metrics (e.g., RMSE, MAE) for each model and target.
4) Runs Diebold–Mariano tests comparing each model to a chosen benchmark, using a
   Newey–West HAC variance estimator to account for serial correlation and
   overlapping multi-step forecast errors.

Inputs
- Forecast CSVs written by 04b, 04c, and 04d (one file per horizon).

Outputs (per horizon)
- merged_forecasts_h{h}.csv : realized values and all model predictions
- merged_metrics_h{h}.csv   : evaluation metrics for each model
- merged_dm_tests_h{h}.csv  : DM test results versus the benchmark
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


# ------------------------------------------------------------------------------
# Helper functions (paths, alignment utilities, and statistical tests)
# ------------------------------------------------------------------------------
# =========================================================
# Project paths helper (portable, no hardcoded Desktop paths)
# =========================================================
from pathlib import Path

def get_project_root(start: Path | None = None) -> Path:
    """Find the project (repo) root regardless of the current working directory.

    Strategy: walk upward from `start` (defaults to this script) until we find a
    directory containing a `Data/` folder, or a repo marker like `.git`.
    """
    if start is None:
        start = Path(__file__).resolve()
    start = start.resolve()
    if start.is_file():
        start = start.parent

    markers = {".git", "pyproject.toml", "requirements.txt", "environment.yml"}
    for p in [start] + list(start.parents):
        if (p / "Data").is_dir():
            return p
        if any((p / m).exists() for m in markers):
            return p
    return start

def ensure_data_tree(root: Path) -> dict[str, Path]:
    """Create the standard Data/ subfolders (idempotent)."""
    data = root / "Data"
    raw = data / "Raw data"
    processed = data / "Processed data"
    validation = data / "Data validation"
    oos = data / "Pseudo OOS Forecasting Results"
    for p in (data, raw, processed, validation, oos):
        p.mkdir(parents=True, exist_ok=True)
    return {"root": root, "data": data, "raw": raw, "processed": processed, "validation": validation, "oos": oos}

PROJECT_ROOT = get_project_root()
PATHS = ensure_data_tree(PROJECT_ROOT)
DATA_DIR = PATHS["data"]

def resolve_out_dir() -> Path:
    here = Path(__file__).resolve().parent
    candidates = [here] + list(here.parents)
    for c in candidates:
        p = c / "Data" / "Pseudo OOS Forecasting Results"
        if p.exists():
            return p
    # If the script is run from an unexpected working directory, fall back to a standard relative path.
    p = Path.cwd() / "Data" / "Pseudo OOS Forecasting Results"
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_horizons(out_dir: Path) -> Tuple[int, ...]:
    hs: set[int] = set()

    # Preferred source for the lag length: lag_config.json written by the baseline step (if available).
    p = out_dir / "lag_config.json"
    if p.exists():
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
            for x in obj.get("horizons", []):
                hs.add(int(x))
        except Exception:
            pass

    # Otherwise infer available horizons by scanning the forecast output directory for known filenames.
    pat = re.compile(r"_(?:forecasts|metrics|dm_vs_var|dm_table)_h(\d+)\.csv$")
    for f in out_dir.glob("*.csv"):
        m = pat.search(f.name)
        if m:
            hs.add(int(m.group(1)))

    # If no horizons can be inferred, use a small default set so the pipeline still runs end-to-end.
    if not hs:
        hs = {1, 12}

    return tuple(sorted(hs))


def load_one(out_dir: Path, stem: str, h: int) -> pd.DataFrame:
    p = out_dir / f"{stem}_h{h}.csv"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p)
    if "origin_date" in df.columns:
        df["origin_date"] = pd.to_datetime(df["origin_date"])
    if "horizon_months" in df.columns:
        df["horizon_months"] = df["horizon_months"].astype(int)
    return df


def directional_accuracy(y_true: np.ndarray, y_hat: np.ndarray) -> float:
    # Approximate direction changes using first differences of the realized series (simple turning-point diagnostic).
    # If first differences are unavailable (e.g., too short), classify direction using levels relative to the median.
    if len(y_true) < 3:
        return float("nan")
    dy = np.diff(y_true)
    dh = np.diff(y_hat)
    m = np.isfinite(dy) & np.isfinite(dh)
    if m.sum() == 0:
        return float("nan")
    return float(np.mean(np.sign(dy[m]) == np.sign(dh[m])))


def turning_point_f1(y_true: np.ndarray, y_hat: np.ndarray) -> float:
    # Use the sign of the second difference as a coarse proxy for acceleration/deceleration (diagnostic only).
    if len(y_true) < 5:
        return float("nan")
    d2y = np.diff(y_true, n=2)
    d2h = np.diff(y_hat, n=2)
    m = np.isfinite(d2y) & np.isfinite(d2h)
    if m.sum() == 0:
        return float("nan")
    yt = (np.sign(d2y[m]) > 0).astype(int)
    yh = (np.sign(d2h[m]) > 0).astype(int)

    tp = np.sum((yt == 1) & (yh == 1))
    fp = np.sum((yt == 0) & (yh == 1))
    fn = np.sum((yt == 1) & (yh == 0))
    if tp == 0:
        return 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if (prec + rec) == 0:
        return 0.0
    return float(2 * prec * rec / (prec + rec))


# ------------------------------------------------------------------------------
# Forecast evaluation: metrics and Diebold–Mariano accuracy comparisons
# ------------------------------------------------------------------------------
def compute_metrics(df: pd.DataFrame) -> pd.DataFrame:
    id_cols = {"origin_date", "target", "horizon_months", "y_true"}
    model_cols = [c for c in df.columns if c not in id_cols]
    rows: List[Dict] = []
    for (tgt, h), sub in df.groupby(["target", "horizon_months"]):
        y_true = sub["y_true"].to_numpy(dtype=float)
        for m in model_cols:
            y_hat = sub[m].to_numpy(dtype=float)
            v = np.isfinite(y_true) & np.isfinite(y_hat)
            if v.sum() < 20:
                continue
            yt, yh = y_true[v], y_hat[v]
            rows.append({
                "target": tgt,
                "horizon_months": int(h),
                "model": m,
                "n_obs": int(v.sum()),
                "rmse": float(np.sqrt(np.mean((yh - yt) ** 2))),
                "mae": float(np.mean(np.abs(yh - yt))),
                "directional_accuracy": directional_accuracy(yt, yh),
                "turning_point_f1": turning_point_f1(yt, yh),
            })
    return pd.DataFrame(rows)


def newey_west_var(x: np.ndarray, lag: int) -> float:
    """Estimate var(mean(x)) with Newey–West HAC variance (Bartlett kernel).

Why HAC is needed
- Forecast loss differentials can be serially correlated, especially for horizons
  h > 1 where forecast errors overlap.
- The Newey–West estimator provides a consistent estimate of the long-run
  variance under weak dependence.

Returns
- An estimate of the variance of the sample mean of x.
"""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    T = len(x)
    if T < 5:
        return float("nan")
    x = x - x.mean()  # mean-correct for autocov estimation (this is OK here)
    gamma0 = np.sum(x * x) / T
    s = gamma0
    L = max(0, int(lag))
    for k in range(1, L + 1):
        w = 1.0 - k / (L + 1.0)
        gk = np.sum(x[k:] * x[:-k]) / T
        s += 2.0 * w * gk
    # Convert the estimated long-run variance of x into the variance of the sample mean (divide by T).
    return float(s / T)


def dm_test(y_true: np.ndarray, y_bench: np.ndarray, y_model: np.ndarray, h: int, hlm: bool = True) -> Tuple[float, float]:
    """Diebold–Mariano test for equal predictive accuracy against a benchmark.

Test object
- Let d_t be the loss differential between a candidate model and the benchmark
  (here, squared error loss).
- The null hypothesis is E[d_t] = 0, meaning both models have equal expected loss.

Variance estimation
- Uses a Newey–West HAC estimator with a lag length chosen to reflect the
  forecasting horizon (to handle overlap in multi-step errors).

Returns
- DM statistic and a two-sided p-value.
"""
    y_true = np.asarray(y_true, dtype=float)
    y_bench = np.asarray(y_bench, dtype=float)
    y_model = np.asarray(y_model, dtype=float)
    v = np.isfinite(y_true) & np.isfinite(y_bench) & np.isfinite(y_model)
    yt, yb, ym = y_true[v], y_bench[v], y_model[v]
    T = len(yt)
    if T < 20:
        return (float("nan"), float("nan"))

    e0 = yt - yb
    e1 = yt - ym
    d = (e0 ** 2) - (e1 ** 2)          # positive => model better than benchmark
    dbar = float(np.mean(d))           # DO NOT demean before computing mean

    lag = max(h - 1, 1)
    var_mean = newey_west_var(d, lag=lag)
    if not np.isfinite(var_mean) or var_mean <= 0:
        return (float("nan"), float("nan"))

    dm = dbar / math.sqrt(var_mean)

    # Apply the Harvey–Leybourne–Newbold small-sample correction to the DM statistic.
    if hlm:
        adj = math.sqrt((T + 1 - 2*h + (h*(h-1))/T) / T)
        dm = dm * adj

    # Compute the p-value using the standard normal approximation.
    from math import erf, sqrt
    # Report a two-sided p-value (evidence that either model could be better).
    p = 2.0 * (1.0 - 0.5 * (1.0 + erf(abs(dm) / sqrt(2.0))))
    return (float(dm), float(p))


def compute_dm_vs_var(df: pd.DataFrame, h: int) -> pd.DataFrame:
    rows: List[Dict] = []
    for tgt, sub in df.groupby("target"):
        if "VAR" not in sub.columns:
            continue
        y_true = sub["y_true"].to_numpy(dtype=float)
        y_var = sub["VAR"].to_numpy(dtype=float)

        for m in [c for c in sub.columns if c not in {"origin_date","target","horizon_months","y_true","VAR"}]:
            y_m = sub[m].to_numpy(dtype=float)
            dm, p = dm_test(y_true, y_var, y_m, h=h, hlm=True)
            rows.append({
                "target": tgt,
                "horizon_months": int(h),
                "model": m,
                "benchmark": "VAR",
                "dm_stat": dm,
                "p_value": p,
            })
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------
# Merge forecast files into unified evaluation tables
# ------------------------------------------------------------------------------
def merge_for_h(out_dir: Path, h: int) -> pd.DataFrame:
    base = load_one(out_dir, "baseline_forecasts", h)
    if base.empty:
        return pd.DataFrame()

    xgb = load_one(out_dir, "xgb_forecasts", h)
    rnn = load_one(out_dir, "rnn_forecasts", h)

    df = base.copy()

    # Merge additional models, keeping one y_true
    if not xgb.empty:
        df = df.merge(
            xgb.drop(columns=["y_true"], errors="ignore"),
            on=["origin_date", "target", "horizon_months"],
            how="left",
        )
    if not rnn.empty:
        df = df.merge(
            rnn.drop(columns=["y_true"], errors="ignore"),
            on=["origin_date", "target", "horizon_months"],
            how="left",
        )

    # Ensure sorting
    df = df.sort_values(["target", "origin_date"]).reset_index(drop=True)
    return df


def main() -> None:
    out_dir = resolve_out_dir()
    horizons = read_horizons(out_dir)

    all_fc, all_met, all_dm = [], [], []

    for h in horizons:
        df = merge_for_h(out_dir, h)
        if df.empty:
            continue

        # Save merged forecasts
        df.to_csv(out_dir / f"final_forecasts_h{h}.csv", index=False)

        # Metrics
        met = compute_metrics(df)
        met.to_csv(out_dir / f"final_metrics_h{h}.csv", index=False)

        # DM
        dm = compute_dm_vs_var(df, h=h)
        dm.to_csv(out_dir / f"final_dm_table_h{h}.csv", index=False)

        all_fc.append(df)
        all_met.append(met)
        all_dm.append(dm)

    if not all_fc:
        raise RuntimeError("No horizons processed. Check that baseline_forecasts_h*.csv exist.")

    # Export pooled tables across horizons
    fc_all = pd.concat(all_fc, ignore_index=True)
    met_all = pd.concat(all_met, ignore_index=True)
    dm_all = pd.concat(all_dm, ignore_index=True)

    fc_all.to_csv(out_dir / "final_forecasts_all_horizons.csv", index=False)
    met_all.to_csv(out_dir / "final_metrics_all_horizons.csv", index=False)
    dm_all.to_csv(out_dir / "final_dm_table_all_horizons.csv", index=False)

    # Also save per-horizon files for backward compatibility with plotting script 04f
    for h in sorted(fc_all["horizon_months"].dropna().unique().astype(int).tolist()):
        sub_fc = fc_all.loc[fc_all["horizon_months"] == int(h)].copy()
        sub_met = met_all.loc[met_all["horizon_months"] == int(h)].copy()
        sub_dm = dm_all.loc[dm_all["horizon_months"] == int(h)].copy() if "horizon_months" in dm_all.columns else dm_all.copy()
        sub_fc.to_csv(out_dir / f"final_forecasts_h{int(h)}.csv", index=False)
        sub_met.to_csv(out_dir / f"final_metrics_h{int(h)}.csv", index=False)
        sub_dm.to_csv(out_dir / f"final_dm_table_h{int(h)}.csv", index=False)


    print(f"Wrote final outputs to: {out_dir}")
    print(f"Horizons processed: {horizons}")


if __name__ == "__main__":
    main()
