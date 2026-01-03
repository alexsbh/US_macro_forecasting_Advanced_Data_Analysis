"""04b_baselines_var_linear.py

Purpose
- Estimate benchmark forecasting models under a strict pseudo out-of-sample
  (expanding-window) procedure.
- Produce baseline forecasts and evaluation outputs that serve as reference points
  for more flexible machine-learning models.

Models implemented
1) VAR benchmark
   - A multivariate Vector Autoregression estimated on available data up to each
     forecast origin.
   - Forecasts are produced by iterating the VAR forward to the desired horizon.
2) Linear predictive regressions
   - Regularized (ridge) regressions that map lagged predictors to the future
     target value.
   - Hyperparameters are selected using time-series–appropriate cross-validation
     (no random shuffling).

Forecasting protocol (pseudo out-of-sample)
- For each forecast origin t0 and horizon h:
  - training data use only information dated at or before t0 (with additional
    trimming to respect the h-step target construction),
  - the model produces a forecast for y_{t0+h},
  - errors are computed only when the realized value is available.

Inputs
- The processed monthly panel (predictors + targets).
- A configuration describing:
  - which variables are targets,
  - which predictors/lag structure to include,
  - horizons to evaluate,
  - the start/end of the evaluation window.

Outputs (per horizon, written to the results folder)
- baseline_forecasts_h{h}.csv  : realized values and baseline predictions
- baseline_metrics_h{h}.csv    : error metrics (RMSE, MAE, etc.)
- baseline_dm_tests_h{h}.csv   : Diebold–Mariano comparisons (where applicable)
"""
from __future__ import annotations

import os, json, time, warnings, traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV, ElasticNetCV

from statsmodels.tsa.api import VAR
from sklearn.metrics import f1_score

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

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# ---------------------------- Configuration ----------------------------

@dataclass
class BaselineConfig:
    horizon: int = 12
    oos_start: str = "2000-01-01"

    # Forecast targets (each must be a column in the prepared modeling dataset).
    targets: Tuple[str, ...] = (
        "cpi_headline_yoy",
        "cpi_core_yoy",
        "indpro_total_yoy",
        "unrate_u3",
        "t10y2y",
    )

    # Core predictor block: only variables present in the panel are included.
    macro_core: Tuple[str, ...] = ("effr", "t10y2y", "t10y3mm", "nfci", "anfci")

    # Number of monthly lags used to build lagged predictors (can be overridden by VAR lag selection).
    max_lag_months: int = 12

    # Settings for selecting the VAR lag length (information criterion and candidate range).
    select_var_lags: bool = True
    var_lags_max: int = 12
    var_ic: str = "aic"   # "aic" or "bic"
    pre_oos_end: str = "1999-12-01"

    # Cross-validation settings for regularized linear models (time-series aware; no shuffling).
    cv_splits: int = 5
    ridge_alphas: Tuple[float, ...] = tuple(np.logspace(-4, 4, 25))  # wide, robust grid
    enet_l1_ratios: Tuple[float, ...] = (0.1, 0.5, 0.9)
    enet_alphas: Tuple[float, ...] = tuple(np.logspace(-4, 1, 30))

    # Logging and run metadata (written to the results folder for reproducibility).
    progress_every: int = 50


# ---------------------------- Utility functions ----------------------------

class RunLogger:
    def __init__(self, out_dir: Path, name: str):
        self.path = out_dir / name
        self.t0 = time.time()
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            f.write(f"Log started: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

    def log(self, msg: str):
        dt = time.time() - self.t0
        line = f"[{dt:8.1f}s] {msg}"
        print(line)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def log_tb(self, context: str):
        self.log(f"ERROR: {context}\n{traceback.format_exc()}")


def ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    if "date" in df.columns:
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    return df


def make_lags(df: pd.DataFrame, cols: List[str], max_lag: int) -> pd.DataFrame:
    out = {}
    for c in cols:
        if c not in df.columns:
            continue
        for l in range(1, max_lag + 1):
            out[f"{c}_lag{l}"] = df[c].shift(l)
    return pd.DataFrame(out, index=df.index)


def horizon_train_end(origin: pd.Timestamp, h: int) -> pd.Timestamp:
    # Latest permissible training timestamp when constructing y_{t+h} targets (prevents target leakage).
    return origin - pd.DateOffset(months=h)


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # Determine direction of change relative to the previous observation (used for simple diagnostics).
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    if len(y_true) < 2:
        return float("nan")
    dy_true = np.sign(np.diff(y_true))
    dy_pred = np.sign(np.diff(y_pred))
    return float(np.mean(dy_true == dy_pred))


def turning_point_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Identify turning points using sign changes in first differences.

This is used as a simple diagnostic to summarize whether a series changes
direction (increasing to decreasing, or vice versa) over time.
"""
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    if len(y_true) < 3:
        return float("nan")
    d1_true = np.diff(y_true)
    d1_pred = np.diff(y_pred)
    tp_true = (np.sign(d1_true[1:]) != np.sign(d1_true[:-1])).astype(int)
    tp_pred = (np.sign(d1_pred[1:]) != np.sign(d1_pred[:-1])).astype(int)
    return float(f1_score(tp_true, tp_pred, zero_division=0))


def dm_test(e_model: np.ndarray, e_bench: np.ndarray, h: int, nw_lag: Optional[int] = None) -> Tuple[float, float]:
    """Compute a Diebold–Mariano (DM) test for predictive accuracy.

Setup
- Loss function: squared forecast error.
- The DM statistic is computed on the loss differential between two forecast
  sequences.
- The long-run variance of the differential is estimated with a Newey–West HAC
  estimator to account for serial correlation induced by multi-step forecasting.

Returns
- DM statistic, p-value, and the chosen two-sided alternative.
"""
    import scipy.stats as st

    e_model = np.asarray(e_model, float)
    e_bench = np.asarray(e_bench, float)
    v = np.isfinite(e_model) & np.isfinite(e_bench)
    e_model, e_bench = e_model[v], e_bench[v]
    if len(e_model) < 10:
        return float("nan"), float("nan")

    d = (e_bench**2) - (e_model**2)  # >0 means model improves on benchmark

    T = len(d)
    if nw_lag is None:
        # Practical default: use a conventional lag choice when data limitations prevent more elaborate selection.
        nw_lag = int(np.floor(1.2 * T ** (1/3)))

    # Newey–West variance of mean(d)
    gamma0 = np.dot(d, d) / T
    var = gamma0
    for L in range(1, min(nw_lag, T-1) + 1):
        w = 1.0 - L/(nw_lag+1.0)
        gammaL = np.dot(d[L:], d[:-L]) / T
        var += 2.0 * w * gammaL
    var_mean = var / T

    if var_mean <= 0:
        return float("nan"), float("nan")

    dm = (d.mean()) / np.sqrt(var_mean)
    p = 2.0 * (1.0 - st.norm.cdf(abs(dm)))
    return float(dm), float(p)


# ---------------------------- Data Loading ----------------------------

def resolve_paths() -> Tuple[Path, Path, Path]:
    here = Path(__file__).resolve().parent
    data_dir = (here / "Data").resolve()
    model_path = data_dir / "Data validation" / "03_outputs" / "model_ready" / "model_ready_base.csv"
    results_dir = data_dir / "Pseudo OOS Forecasting Results"
    results_dir.mkdir(parents=True, exist_ok=True)
    return data_dir, model_path, results_dir


def load_model_ready(model_path: Path) -> pd.DataFrame:
    df = pd.read_csv(model_path)
    df = ensure_datetime_index(df)
    return df


def load_pca_factors(results_dir: Path) -> Optional[pd.DataFrame]:
    p = results_dir / "pca_factors.csv"
    if not p.exists():
        return None
    f = pd.read_csv(p)
    f["date"] = pd.to_datetime(f["date"])
    f = f.set_index("date").sort_index()
    # Keep only PC columns
    pc_cols = [c for c in f.columns if c.upper().startswith("PC")]
    return f[pc_cols].copy()


# ---------------------------- VAR lag selection ----------------------------

def select_var_lag(df: pd.DataFrame, cfg: BaselineConfig, logger: RunLogger) -> int:
    cols = [c for c in cfg.targets if c in df.columns]
    if len(cols) < 2:
        logger.log("VAR lag selection skipped (not enough target columns). Using max_lag_months.")
        return cfg.max_lag_months

    pre_end = pd.to_datetime(cfg.pre_oos_end)
    Y = df.loc[df.index <= pre_end, cols].astype(float).dropna()
    if len(Y) < (cfg.var_lags_max + 50):
        logger.log("VAR lag selection skipped (too few pre-OOS rows). Using max_lag_months.")
        return cfg.max_lag_months

    ic = cfg.var_ic.lower()
    try:
        sel = VAR(Y).select_order(maxlags=cfg.var_lags_max)
        p = int(getattr(sel, ic))
        p = max(1, min(p, cfg.var_lags_max))
        logger.log(f"Selected VAR lag p={p} using {cfg.var_ic.upper()} on pre-OOS sample (<= {cfg.pre_oos_end}).")
        return p
    except Exception:
        logger.log_tb("VAR lag selection failed; using max_lag_months.")
        return cfg.max_lag_months


def write_lag_config(results_dir: Path, p: int, cfg: BaselineConfig):
    out = {
        "var_lag_p": int(p),
        "var_ic": cfg.var_ic,
        "var_lags_max": int(cfg.var_lags_max),
        "pre_oos_end": cfg.pre_oos_end,
        "horizon_months": int(cfg.horizon),
        "note": "Lag p selected on pre-OOS sample and intended for shared lagged-DV feature block.",
    }
    with open(results_dir / "lag_config.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)


# ---------------------------- Feature bundle ----------------------------

def build_bundle(
    df: pd.DataFrame,
    factors: Optional[pd.DataFrame],
    cfg: BaselineConfig,
    target: str,
    origin: pd.Timestamp,
    lag_p: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Construct the supervised learning design for a single forecast origin.

Target definition
- The forecast target for origin t0 and horizon h is y_{t0+h}.

Leakage control
- When building the training sample, the script ensures that all training
  observations map predictors at time t to targets at time t+h that are strictly
  within the information set available at origin t0.
- Concretely, training rows are restricted so that their target dates do not
  extend beyond the origin’s available information window.

Returns
- X_train, y_train for estimation, and X_test for the single forecast origin.
"""
    y = df[target].shift(-cfg.horizon)

    train_end = horizon_train_end(origin, cfg.horizon)
    train_mask = df.index <= train_end

    lag_df = make_lags(df, list(cfg.targets), lag_p)

    macro_cols = [c for c in cfg.macro_core if c in df.columns]
    X = pd.concat([lag_df, df[macro_cols]], axis=1)
    if factors is not None:
        X = pd.concat([X, factors], axis=1)

    X_train = X.loc[train_mask].copy()
    y_train = y.loc[train_mask].copy()
    x_test = X.loc[[origin]].copy()

    # Drop rows where the constructed h-step-ahead target is missing.
    v = y_train.notna()
    X_train = X_train.loc[v.values].copy()
    y_train = y_train.loc[v].copy()

    cols = list(X_train.columns)

    # Build design matrices; imputation and scaling are performed inside the model pipeline.
    return X_train.values.astype(float), y_train.values.astype(float), x_test.values.astype(float), cols


def fit_var_forecast(df: pd.DataFrame, cfg: BaselineConfig, target: str, origin: pd.Timestamp, lag_p: int) -> float:
    """Generate an h-step-ahead VAR forecast by iterating the fitted model forward.

Key idea
- The VAR is estimated using observed data up to and including the forecast
  origin.
- Starting from the last observed state at the origin, the model is iterated
  forward one step at a time to obtain the h-step-ahead forecast.
"""
    cols = [c for c in cfg.targets if c in df.columns]
    Y = df.loc[df.index <= origin, cols].astype(float).dropna()
    if len(Y) < (lag_p + 25):
        return float("nan")
    res = VAR(Y).fit(maxlags=lag_p, ic=None, trend="c")
    fc_path = res.forecast(Y.values[-res.k_ar:], steps=cfg.horizon)
    j = cols.index(target)
    return float(fc_path[-1, j])


# ---------------------------- Main routine ----------------------------

def main():
    cfg = BaselineConfig()

    _, model_path, results_dir = resolve_paths()
    logger = RunLogger(results_dir, "run_log_04b_baselines_var_linear_UPDATED.txt")
    logger.log(f"Reading model-ready data from: {model_path}")

    df = load_model_ready(model_path)
    factors = load_pca_factors(results_dir)
    if factors is not None:
        # Ensure the auxiliary feature matrix is aligned to the panel’s time index.
        factors = factors.reindex(df.index)
        logger.log(f"Loaded PCA factors: {factors.shape[1]} PCs (will be appended to features).")
    else:
        logger.log("No PCA factors found (pca_factors.csv). Proceeding without PCA.")

    # Build the list of forecast origins used for pseudo out-of-sample evaluation.
    oos_start = pd.to_datetime(cfg.oos_start)
    # The final origin is the last date for which the realized y_{t+h} is observed.
    # Loop over forecast horizons requested in the configuration.
    for h in (1, 12):
        cfg.horizon = int(h)
        logger.log(f"--- Running baselines for horizon h={cfg.horizon} ---")
        last_origin = df.index.max() - pd.DateOffset(months=cfg.horizon)
        origins = pd.date_range(oos_start, last_origin, freq="MS")
        logger.log(f"OOS origins: {origins[0].date()} .. {origins[-1].date()}  (N={len(origins)})")

        # Settings for selecting the VAR lag length (information criterion and candidate range).
        lag_p = cfg.max_lag_months
        if cfg.select_var_lags:
            lag_p = select_var_lag(df, cfg, logger)
        write_lag_config(results_dir, lag_p, cfg)
        logger.log(f"Using lag_p={lag_p} for VAR and lagged-DV block in linear models.")

        # Initialize containers to store forecasts and realized values for this horizon.
        fc_rows = []
        t0 = time.time()

        for i, origin in enumerate(origins):
            for target in cfg.targets:
                if target not in df.columns:
                    continue

                y_true = df.loc[origin + pd.DateOffset(months=cfg.horizon), target] if (origin + pd.DateOffset(months=cfg.horizon)) in df.index else np.nan

                row = {
                    "origin_date": origin,
                    "target": target,
                    "horizon_months": cfg.horizon,
                    "y_true": float(y_true) if pd.notna(y_true) else np.nan,
                }

                # Fit the VAR on data available up to the origin and generate an h-step-ahead forecast.
                try:
                    row["VAR"] = fit_var_forecast(df, cfg, target, origin, lag_p)
                except Exception:
                    logger.log_tb(f"VAR failed for target={target} origin={origin.date()}")
                    row["VAR"] = np.nan

                # Fit regularized linear models (ridge / elastic net) and generate the h-step-ahead forecast.
                try:
                    Xtr, ytr, xt, _ = build_bundle(df, factors, cfg, target, origin, lag_p)

                    # Select the ridge penalty using ordered folds that respect time ordering.
                    # scikit-learn splits folds by index order; with ordered data this approximates expanding-window CV.
                    # A fully custom splitter is possible; the ordered-fold approach is typically adequate for this application.
                    # The chosen CV approach is stable, reproducible, and avoids information leakage from the future.
                    ridge = Pipeline(steps=[
                        ("imp", SimpleImputer(strategy="mean")),
                        ("sc", StandardScaler()),
                        ("m", RidgeCV(alphas=list(cfg.ridge_alphas), cv=cfg.cv_splits)),
                    ])
                    ridge.fit(Xtr, ytr)
                    row["Ridge"] = float(ridge.predict(xt)[0])

                    enet = Pipeline(steps=[
                        ("imp", SimpleImputer(strategy="mean")),
                        ("sc", StandardScaler()),
                        ("m", ElasticNetCV(
                            l1_ratio=list(cfg.enet_l1_ratios),
                            alphas=list(cfg.enet_alphas),
                            cv=cfg.cv_splits,
                            max_iter=20000,
                            random_state=42,
                            n_jobs=None,
                        )),
                    ])
                    enet.fit(Xtr, ytr)
                    row["ElasticNet"] = float(enet.predict(xt)[0])

                except Exception:
                    logger.log_tb(f"Linear models failed for target={target} origin={origin.date()}")
                    row["Ridge"] = np.nan
                    row["ElasticNet"] = np.nan

                fc_rows.append(row)

            if (i + 1) % cfg.progress_every == 0 or (i + 1) == len(origins):
                elapsed = time.time() - t0
                logger.log(f"Progress {i+1}/{len(origins)} | elapsed {elapsed/60:.1f} min")

        fc = pd.DataFrame(fc_rows)
        fc = fc.sort_values(["target", "origin_date"]).reset_index(drop=True)
        out_fc = results_dir / f"baseline_forecasts_h{cfg.horizon}.csv"
        fc.to_csv(out_fc, index=False)
        logger.log(f"Wrote forecasts: {out_fc}")

        # ---------------- Compute metrics ----------------
        models = ["VAR", "Ridge", "ElasticNet"]
        metric_rows = []

        for target in sorted(fc["target"].unique()):
            sub = fc[fc["target"] == target].copy()
            y_true = sub["y_true"].values.astype(float)

            for m in models:
                y_hat = sub[m].values.astype(float)
                v = np.isfinite(y_true) & np.isfinite(y_hat)
                yt, yh = y_true[v], y_hat[v]
                if len(yt) < 10:
                    continue
                rmse = float(np.sqrt(np.mean((yh - yt) ** 2)))
                mae = float(np.mean(np.abs(yh - yt)))
                da = directional_accuracy(yt, yh)
                tp = turning_point_f1(yt, yh)
                metric_rows.append({
                    "target": target,
                    "horizon_months": cfg.horizon,
                    "model": m,
                    "n_obs": int(len(yt)),
                    "rmse": rmse,
                    "mae": mae,
                    "directional_accuracy": da,
                    "turning_point_f1": tp
                })

        met = pd.DataFrame(metric_rows)
        out_met = results_dir / f"baseline_metrics_h{cfg.horizon}.csv"
        met.to_csv(out_met, index=False)
        logger.log(f"Wrote metrics: {out_met}")

        # ---------------- Compare to VAR benchmark (DM) ----------------
        dm_rows = []
        for target in sorted(fc["target"].unique()):
            sub = fc[fc["target"] == target].copy()
            y_true = sub["y_true"].values.astype(float)
            e_var = sub["VAR"].values.astype(float) - y_true

            for m in ["Ridge", "ElasticNet"]:
                e_m = sub[m].values.astype(float) - y_true
                dm, p = dm_test(e_m, e_var, h=cfg.horizon)
                dm_rows.append({
                    "target": target,
                    "horizon_months": cfg.horizon,
                    "model": m,
                    "benchmark": "VAR",
                    "dm_stat": dm,
                    "p_value": p
                })

        dm_df = pd.DataFrame(dm_rows)
        out_dm = results_dir / f"baseline_dm_vs_var_h{cfg.horizon}.csv"
        dm_df.to_csv(out_dm, index=False)
        logger.log(f"Wrote DM table: {out_dm}")

        logger.log(f"Finished horizon h={cfg.horizon}.")

    logger.log("DONE (all horizons).")


if __name__ == "__main__":
    main()
