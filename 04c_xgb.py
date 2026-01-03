"""04c_xgb.py

Purpose
- Train gradient-boosted tree forecasting models (XGBoost) under the same strict
  pseudo out-of-sample (expanding-window) protocol used for the baselines.

What the script does
1) Loads the processed panel and any auxiliary features (e.g., PCA factors).
2) Builds lagged predictors using a lag length read from the shared configuration
   produced by the baseline step, so all models use a consistent lag structure.
3) For each forecast horizon h and each forecast origin:
   - fits an XGBoost model using only information available up to that origin,
   - generates an h-step-ahead forecast for the target,
   - stores the forecast alongside the realized value.
4) Computes standard forecast accuracy metrics and (optionally) runs
   Diebold–Mariano comparisons against a baseline benchmark.

Inputs
- Processed monthly panel.
- lag_config.json (for the lag length p used to construct lagged features).
- Baseline forecast files (used for alignment and DM testing when enabled).

Outputs (per horizon)
- xgb_forecasts_h{h}.csv   : realized values and XGBoost predictions
- xgb_metrics_h{h}.csv     : error metrics (RMSE, MAE, etc.)
- xgb_dm_vs_var_h{h}.csv   : DM test results versus the VAR benchmark
- run_log_04c_xgb.txt      : run metadata (timestamps, configuration, warnings)
"""
from __future__ import annotations

import json, time, warnings, traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

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

try:
    from xgboost import XGBRegressor
except Exception as e:
    raise RuntimeError("xgboost is required. Install xgboost.") from e


@dataclass
class Config:
    horizons: Tuple[int, ...] = (1, 12)
    oos_start: str = "2000-01-01"

    targets: Tuple[str, ...] = (
        "cpi_headline_yoy",
        "cpi_core_yoy",
        "indpro_total_yoy",
        "unrate_u3",
        "t10y2y",
    )
    macro_core: Tuple[str, ...] = ("effr", "t10y2y", "t10y3mm", "nfci", "anfci")
    progress_every: int = 60

    # Define XGBoost hyperparameters (chosen to be robust for monthly macro data and expanding-window training).
    xgb_params: Dict = None

    def __post_init__(self):
        if self.xgb_params is None:
            self.xgb_params = dict(
                n_estimators=800,
                max_depth=3,
                learning_rate=0.04,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda=1.0,
                min_child_weight=1.0,
                objective="reg:squarederror",
                random_state=42,
                n_jobs=4,
            )


class RunLogger:
    def __init__(self, out_dir: Path, name: str):
        out_dir.mkdir(parents=True, exist_ok=True)
        self.path = out_dir / name
        self.t0 = time.time()
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


def resolve_paths() -> tuple[Path, Path]:
    here = Path(__file__).resolve().parent
    data_path = (here / "Data" / "Data validation" / "03_outputs" / "model_ready" / "model_ready_base.csv").resolve()
    out_dir = (here / "Data" / "Pseudo OOS Forecasting Results").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    return data_path, out_dir


def load_df(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
    df.index = pd.to_datetime(df.index)
    return df.sort_index()


def load_factors(out_dir: Path, index: pd.DatetimeIndex) -> Optional[pd.DataFrame]:
    p = out_dir / "pca_factors.csv"
    if not p.exists():
        return None
    f = pd.read_csv(p)
    f["date"] = pd.to_datetime(f["date"])
    f = f.set_index("date").sort_index()
    pc_cols = [c for c in f.columns if c.upper().startswith("PC")]
    return f[pc_cols].reindex(index)


def read_lag_p(out_dir: Path, fallback: int = 12) -> int:
    p = out_dir / "lag_config.json"
    if not p.exists():
        return fallback
    obj = json.loads(p.read_text(encoding="utf-8"))
    return int(obj.get("var_lag_p", fallback))


def make_lags(df: pd.DataFrame, cols: List[str], p: int) -> pd.DataFrame:
    out = {}
    for c in cols:
        if c not in df.columns:
            continue
        for l in range(1, p+1):
            out[f"{c}_lag{l}"] = df[c].shift(l)
    return pd.DataFrame(out, index=df.index)


def train_end_for_origin(origin: pd.Timestamp, h: int) -> pd.Timestamp:
    return origin - pd.DateOffset(months=h)


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2:
        return float("nan")
    return float(np.mean(np.sign(np.diff(y_true)) == np.sign(np.diff(y_pred))))


def turning_point_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 3:
        return float("nan")
    d_true = np.diff(y_true)
    d_pred = np.diff(y_pred)
    tp_true = (np.sign(d_true[1:]) != np.sign(d_true[:-1])).astype(int)
    tp_pred = (np.sign(d_pred[1:]) != np.sign(d_pred[:-1])).astype(int)
    return float(f1_score(tp_true, tp_pred, zero_division=0))


def dm_test_sqerr(e_model: np.ndarray, e_var: np.ndarray) -> tuple[float, float]:
    import scipy.stats as st
    v = np.isfinite(e_model) & np.isfinite(e_var)
    e_model, e_var = e_model[v], e_var[v]
    if len(e_model) < 20:
        return float("nan"), float("nan")
    d = (e_var**2) - (e_model**2)
    d = d - d.mean()
    T = len(d)
    L = int(np.floor(1.2 * T ** (1/3)))
    gamma0 = np.dot(d, d) / T
    var = gamma0
    for l in range(1, min(L, T-1)+1):
        w = 1.0 - l/(L+1.0)
        gam = np.dot(d[l:], d[:-l]) / T
        var += 2.0 * w * gam
    var_mean = var / T
    if var_mean <= 0:
        return float("nan"), float("nan")
    dm = d.mean() / np.sqrt(var_mean)
    p = 2.0 * (1.0 - st.norm.cdf(abs(dm)))
    return float(dm), float(p)


def build_Xy(df: pd.DataFrame, factors: Optional[pd.DataFrame], cfg: Config, target: str, origin: pd.Timestamp, p: int, h: int):
    y = df[target].shift(-h)
    train_end = train_end_for_origin(origin, h)
    train_mask = df.index <= train_end

    lag_df = make_lags(df, list(cfg.targets), p)
    macro_cols = [c for c in cfg.macro_core if c in df.columns]
    X = pd.concat([lag_df, df[macro_cols]], axis=1)
    if factors is not None:
        X = pd.concat([X, factors], axis=1)

    X_train = X.loc[train_mask]
    y_train = y.loc[train_mask]
    v = y_train.notna()
    X_train = X_train.loc[v.values]
    y_train = y_train.loc[v]
    x_test = X.loc[[origin]]
    return X_train.values.astype(float), y_train.values.astype(float), x_test.values.astype(float)


def main():
    cfg = Config()
    data_path, out_dir = resolve_paths()
    logger = RunLogger(out_dir, "run_log_04c_xgb.txt")
    logger.log(f"Reading: {data_path}")
    df = load_df(data_path)

    factors = load_factors(out_dir, df.index)
    if factors is not None:
        logger.log(f"Loaded PCA factors: {factors.shape[1]} PCs.")
    else:
        logger.log("No PCA factors found; proceeding without PCA.")

    p = read_lag_p(out_dir, fallback=12)
    logger.log(f"Using lag_p={p} for lagged-DV feature block.")

    targets = [c for c in cfg.targets if c in df.columns]

    for h in cfg.horizons:
        # Load baseline (VAR) errors so forecast accuracy can be compared using a Diebold–Mariano test.
        base_fc_path = out_dir / f"baseline_forecasts_h{h}.csv"
        if not base_fc_path.exists():
            raise FileNotFoundError(f"Missing baseline forecasts for h={h}: {base_fc_path.name} (run 04b first)")
        base_fc = pd.read_csv(base_fc_path)
        base_fc["origin_date"] = pd.to_datetime(base_fc["origin_date"])

        oos_start = pd.to_datetime(cfg.oos_start)
        last_origin = df.index.max() - pd.DateOffset(months=h)
        origins = pd.date_range(oos_start, last_origin, freq="MS")

        logger.log(f"--- Horizon h={h} --- origins={len(origins)}")

        t_loop = time.time()
        rows = []

        for i, origin in enumerate(origins):
            for tgt in targets:
                t_true = origin + pd.DateOffset(months=h)
                y_true = df.loc[t_true, tgt] if t_true in df.index else np.nan

                r = {"origin_date": origin, "target": tgt, "horizon_months": h, "y_true": float(y_true) if pd.notna(y_true) else np.nan}

                try:
                    Xtr, ytr, xt = build_Xy(df, factors, cfg, tgt, origin, p, h)

                    model = Pipeline([
                        ("imp", SimpleImputer(strategy="mean")),
                        ("sc", StandardScaler()),
                        ("m", XGBRegressor(**cfg.xgb_params)),
                    ])
                    model.fit(Xtr, ytr)
                    r["XGBoost"] = float(model.predict(xt)[0])
                except Exception:
                    logger.log_tb(f"XGB failed | h={h} tgt={tgt} origin={origin.date()}")
                    r["XGBoost"] = np.nan

                rows.append(r)

            if (i + 1) % cfg.progress_every == 0 or (i + 1) == len(origins):
                logger.log(f"h={h} progress {i+1}/{len(origins)} elapsed={((time.time()-t_loop)/60):.1f} min")

        fc = pd.DataFrame(rows).sort_values(["target", "origin_date"]).reset_index(drop=True)
        fc_path = out_dir / f"xgb_forecasts_h{h}.csv"
        fc.to_csv(fc_path, index=False)
        logger.log(f"Wrote {fc_path.name}")

        # Compute and store forecast accuracy metrics for this horizon (e.g., RMSE, MAE).
        mrows = []
        for tgt in sorted(fc["target"].unique()):
            sub = fc[fc["target"] == tgt]
            yt = sub["y_true"].values.astype(float)
            yh = sub["XGBoost"].values.astype(float)
            v = np.isfinite(yt) & np.isfinite(yh)
            if v.sum() < 20:
                continue
            ytv, yhv = yt[v], yh[v]
            rmse = float(np.sqrt(np.mean((yhv - ytv)**2)))
            mae = float(np.mean(np.abs(yhv - ytv)))
            da = directional_accuracy(ytv, yhv)
            tp = turning_point_f1(ytv, yhv)
            mrows.append({"target": tgt, "horizon_months": h, "model": "XGBoost", "n_obs": int(v.sum()),
                          "rmse": rmse, "mae": mae, "directional_accuracy": da, "turning_point_f1": tp})
        met = pd.DataFrame(mrows)
        met_path = out_dir / f"xgb_metrics_h{h}.csv"
        met.to_csv(met_path, index=False)
        logger.log(f"Wrote {met_path.name}")

        # Run Diebold–Mariano tests versus the VAR benchmark using squared-error loss.
        drows = []
        merged = base_fc.merge(fc[["origin_date","target","horizon_months","XGBoost"]],
                               on=["origin_date","target","horizon_months"], how="inner")
        for tgt in sorted(merged["target"].unique()):
            sub = merged[merged["target"] == tgt]
            yt = sub["y_true"].values.astype(float)
            e_var = sub["VAR"].values.astype(float) - yt
            e_m = sub["XGBoost"].values.astype(float) - yt
            dm, pv = dm_test_sqerr(e_m, e_var)
            drows.append({"target": tgt, "horizon_months": h, "model": "XGBoost", "benchmark": "VAR", "dm_stat": dm, "p_value": pv})
        dm = pd.DataFrame(drows)
        dm_path = out_dir / f"xgb_dm_vs_var_h{h}.csv"
        dm.to_csv(dm_path, index=False)
        logger.log(f"Wrote {dm_path.name}")

    logger.log("DONE.")


if __name__ == "__main__":
    main()
