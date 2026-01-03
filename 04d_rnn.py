"""04d_rnn.py

Purpose
- Train recurrent neural network forecasting models (LSTM and GRU) under a strict
  pseudo out-of-sample expanding-window setup.

Why sequence models
- Unlike linear models and tree ensembles that rely on explicit lags, RNNs can
  learn temporal dependencies directly from sequences of past observations.
- This script constructs fixed-length input sequences and predicts the target at
  a specified future horizon.

Forecasting protocol
- For each horizon h and each forecast origin t0:
  - the training window includes only data available up to the origin,
  - sequences are constructed from the predictor panel (lags + contemporaneous
    features),
  - the model is trained on the training window and then used to forecast
    y_{t0+h}.

Model variants
- LSTM and GRU architectures with standard training setups.
- Optimization settings (learning rate, epochs, batch size) are defined in the
  configuration section of the script.

Inputs
- Processed monthly panel.
- Optional PCA factor file produced by the PCA step.
- Shared lag configuration for consistent lag length across models.

Outputs (per horizon)
- rnn_forecasts_h{h}.csv   : realized values and RNN predictions
- rnn_metrics_h{h}.csv     : error metrics
- run_log_04d_rnn.txt      : run metadata and training diagnostics
"""
from __future__ import annotations
import argparse
from sklearn.pipeline import Pipeline

import json, time, warnings, traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
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
    import tensorflow as tf
    from tensorflow.keras import layers, models, callbacks, optimizers
except Exception as e:
    raise RuntimeError("tensorflow is required for 04d_rnn.py") from e


@dataclass
class Config:
    horizons: Tuple[int, ...] = (1, 12)
    # Choose which sequence-model architectures to estimate (LSTM and/or GRU).
    model_types: Tuple[str, ...] = ("LSTM", "GRU")
    oos_start: str = "2000-01-01"

    targets: Tuple[str, ...] = (
        "cpi_headline_yoy",
        "cpi_core_yoy",
        "indpro_total_yoy",
        "unrate_u3",
        "t10y2y",
    )
    macro_core: Tuple[str, ...] = ("effr", "t10y2y", "t10y3mm", "nfci", "anfci")

    # Length of each input sequence, in months (number of past observations fed into the network).
    seq_len: int = 12

    # Retrain frequency: how often the network is refit as the expanding window moves forward.
    retrain_every: int = 24

    # Training hyperparameters (epochs, batch size, early stopping settings if used).
    max_epochs: int = 25
    batch_size: int = 32
    patience: int = 10

    # Network architecture hyperparameters (hidden units, layers, dropout).
    rnn_units: int = 32
    dense_units: int = 16
    dropout: float = 0.1

    # Optimizer configurations (learning rate, momentum/Adam parameters).
    use_adam: bool = True
    use_sgd: bool = True   # default ON to produce 4 series (LSTM/GRU × Adam/SGD)
    sgd_lr: float = 0.01
    adam_lr: float = 0.001

    progress_every: int = 24  # origins


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


def train_end_for_origin(origin: pd.Timestamp, h: int) -> pd.Timestamp:
    return origin - pd.DateOffset(months=h)


def make_lags(df: pd.DataFrame, cols: List[str], p: int) -> pd.DataFrame:
    out = {}
    for c in cols:
        if c not in df.columns:
            continue
        for l in range(1, p+1):
            out[f"{c}_lag{l}"] = df[c].shift(l)
    return pd.DataFrame(out, index=df.index)


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


def build_feature_frame(df: pd.DataFrame, factors: Optional[pd.DataFrame], cfg: Config, p: int) -> pd.DataFrame:
    """Create the per-period feature matrix used to build sequences.

Included information at each time t
- Lagged values of the target (lags 1..p) to encode recent dynamics.
- Contemporaneous core macro predictors (levels or transformations as defined in
  the processed panel).
- Optional contemporaneous PCA factors (if provided), which summarize common
  variation across many predictors.

The resulting matrix has shape (T, d), where T is the number of time periods and
d is the number of features.
"""
    lag_df = make_lags(df, list(cfg.targets), p)
    macro_cols = [c for c in cfg.macro_core if c in df.columns]
    X = pd.concat([lag_df, df[macro_cols]], axis=1)
    if factors is not None:
        X = pd.concat([X, factors], axis=1)
    return X


def build_sequences(X: np.ndarray, y: np.ndarray, seq_len: int) -> tuple[np.ndarray, np.ndarray]:
    """Convert a time-indexed feature matrix into supervised learning sequences.

Inputs
- X aligned by time: shape (T, d)
- y aligned by time: shape (T,)
- seq_len: number of past periods included in each input sequence

Outputs
- X_seq: shape (N, seq_len, d), where each row is a rolling window of features
- y_seq: shape (N,), where each label corresponds to the target aligned to the
  last timestamp of its input sequence (so horizons can be handled cleanly)

Alignment requirement
- y must use the same time index as X so that the label associated with each
  sequence is unambiguous.
"""
    T, d = X.shape
    if T <= seq_len:
        return np.empty((0, seq_len, d)), np.empty((0,))
    Xs, ys = [], []
    for t in range(seq_len-1, T):
        Xs.append(X[t-seq_len+1:t+1, :])
        ys.append(y[t])
    return np.asarray(Xs, dtype=np.float32), np.asarray(ys, dtype=np.float32)


def make_model(model_type: str, input_shape: tuple[int,int], cfg: Config, opt_name: str):
    tf.keras.backend.clear_session()
    inp = layers.Input(shape=input_shape)
    if model_type == "LSTM":
        x = layers.LSTM(cfg.rnn_units, dropout=cfg.dropout, recurrent_dropout=0.0)(inp)
    else:
        x = layers.GRU(cfg.rnn_units, dropout=cfg.dropout, recurrent_dropout=0.0)(inp)
    x = layers.Dense(cfg.dense_units, activation="relu")(x)
    out = layers.Dense(1)(x)
    m = models.Model(inp, out)

    if opt_name.upper() == "ADAM":
        opt = optimizers.Adam(learning_rate=cfg.adam_lr)
    else:
        opt = optimizers.SGD(learning_rate=cfg.sgd_lr, momentum=0.0)

    m.compile(optimizer=opt, loss="mse")
    return m


def main(cfg: Config | None = None):
    cfg = cfg or Config()
    data_path, out_dir = resolve_paths()
    logger = RunLogger(out_dir, "run_log_04d_rnn.txt")
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
    X_df = build_feature_frame(df, factors, cfg, p)

    # Fit the imputer and scaler using only the current expanding training window (prevents look-ahead bias).
    # This ensures no information from future months influences feature scaling or imputation.
    for h in cfg.horizons:
        base_fc_path = out_dir / f"baseline_forecasts_h{h}.csv"
        if not base_fc_path.exists():
            raise FileNotFoundError(f"Missing baseline forecasts for h={h}: run 04b first.")
        base_fc = pd.read_csv(base_fc_path)
        base_fc["origin_date"] = pd.to_datetime(base_fc["origin_date"])

        oos_start = pd.to_datetime(cfg.oos_start)
        last_origin = df.index.max() - pd.DateOffset(months=h)
        origins = pd.date_range(oos_start, last_origin, freq="MS")

        logger.log(f"--- Horizon h={h} --- origins={len(origins)}")

        # Define retraining origins at a fixed step size (every retrain_every months).
        retrain_ix = list(range(0, len(origins), cfg.retrain_every))
        if retrain_ix[-1] != len(origins) - 1:
            retrain_ix.append(len(origins) - 1)

        all_rows = []

        for model_type in cfg.model_types:
            for opt_name in [n for n in ["Adam","SGD"] if (cfg.use_adam and n=="Adam") or (cfg.use_sgd and n=="SGD")]:
                logger.log(f"Model={model_type} Opt={opt_name} | h={h}")

                # Initialize storage for forecasts produced by this model variant.
                preds = {tgt: [] for tgt in targets}
                pred_dates = []

                for r_i, start_idx in enumerate(retrain_ix):
                    origin = origins[start_idx]
                    train_end = train_end_for_origin(origin, h)

                    # Training indices run up to train_end and must include enough history to form seq_len sequences.
                    train_mask = df.index <= train_end

                    # Fit a separate network per target variable to keep the mapping and evaluation transparent.
                    # Build scaled/imputed X for the full train_mask subset, then sequences per target.
                    X_train_raw = X_df.loc[train_mask].copy()
                    # align labels by time t (label at t is y_{t+h})
                    # we'll later cut sequences such that label corresponds to last time step t in sequence
                    # Labels are constructed as the h-step-ahead target value y_{t+h}.
                    # To avoid leakage, the latest training label date is origin - h (so y_{origin} is not used to predict itself).
                    scaler = Pipeline([
                        ("imp", SimpleImputer(strategy="mean")),
                        ("sc", StandardScaler()),
                    ])
                    X_train_scaled = scaler.fit_transform(X_train_raw.values.astype(float))

                    # Forecast a block of origins until the next scheduled retraining point.
                    next_idx = retrain_ix[r_i + 1] if (r_i + 1) < len(retrain_ix) else len(origins)
                    block_origins = origins[start_idx:next_idx]

                    # Precompute scaled features so block forecasts can be produced efficiently and consistently.
                    # Use the scaler fitted on the training window so scaling does not use future information.
                    X_block_raw = X_df.loc[block_origins].values.astype(float)
                    X_block_scaled = scaler.transform(X_block_raw)

                    # Scale the full feature matrix once per retrain; sequences are then built by slicing indices.
                    X_all_scaled = scaler.transform(X_df.values.astype(float))

                    for tgt in targets:
                        y_all = df[tgt].shift(-h).loc[train_mask].values.astype(float)
                        # Remove training sequences whose corresponding label is missing.
                        valid = np.isfinite(y_all)
                        X_tgt = X_train_scaled[valid]
                        y_tgt = y_all[valid]
# Build rolling sequences from the time-indexed feature matrix.
                        Xs, ys = build_sequences(X_tgt, y_tgt, cfg.seq_len)
                        if len(ys) < 50:
                            # If the training window is too short to form sequences, return missing forecasts for this block.
                            for _ in block_origins:
                                preds[tgt].append(np.nan)
                            continue


                        # Instantiate the network and train it on the constructed sequences for this target.
                        input_dim = Xs.shape[-1]
                        m = make_model(model_type, (cfg.seq_len, input_dim), cfg, opt_name)
                        cb = [callbacks.EarlyStopping(monitor='loss', patience=cfg.patience, restore_best_weights=True)]
                        m.fit(Xs, ys, epochs=cfg.max_epochs, batch_size=cfg.batch_size, verbose=0, callbacks=cb)

                        # Predict this whole block in one call (much faster than per-origin predict)
                        idx_ends = [df.index.get_loc(o) for o in block_origins]

                        idx_starts = [ie - (cfg.seq_len - 1) for ie in idx_ends]

                        # Scale ALL X once per retrain (scaler fit on train sample => no leakage)
                        X_all_scaled = scaler.transform(X_df.values.astype(float))

                        X_batch = []
                        batch_valid = []
                        for is_, ie in zip(idx_starts, idx_ends):
                            if is_ < 0 or ie < 0:
                                batch_valid.append(False)
                                X_batch.append(None)
                            else:
                                batch_valid.append(True)
                                X_batch.append(X_all_scaled[is_:ie+1])

                        if any(batch_valid):
                            X_pred = np.stack([x for x, ok in zip(X_batch, batch_valid) if ok], axis=0).astype(np.float32)
                            yhat = m.predict(X_pred, verbose=0, batch_size=cfg.batch_size).reshape(-1)
                        else:
                            yhat = np.array([], dtype=float)

                        j = 0
                        for ok in batch_valid:
                            if not ok:
                                preds[tgt].append(np.nan)
                            else:
                                preds[tgt].append(float(yhat[j]))
                                j += 1

                    pred_dates.extend(list(block_origins))

                    if (r_i+1) % 1 == 0:
                        logger.log(f"h={h} {model_type}-{opt_name} retrain {r_i+1}/{len(retrain_ix)} done (origin={origin.date()})")

                # Build long-form rows
                # pred_dates contains duplicates across targets loop; but we extended each tgt list in same order
                # We'll reconstruct by iterating over targets and positionally indexing
                n_orig = len(pred_dates)
                # pred_dates were extended once per retrain block, but for each block we appended forecasts per target;
                # preds[tgt] aligns to pred_dates.
                for tgt in targets:
                    if len(preds[tgt]) != n_orig:
                        # safety
                        continue
                    for od, yh in zip(pred_dates, preds[tgt]):
                        t_true = od + pd.DateOffset(months=h)
                        y_true = df.loc[t_true, tgt] if t_true in df.index else np.nan
                        all_rows.append({
                            "origin_date": od,
                            "target": tgt,
                            "horizon_months": h,
                            "y_true": float(y_true) if pd.notna(y_true) else np.nan,
                            f"{model_type}_{opt_name}": yh
                        })

        # Convert to wide format: merge columns for each model variant
        fc_long = pd.DataFrame(all_rows)
        if fc_long.empty:
            logger.log(f"No forecasts produced for h={h}.")
            continue

        # Aggregate by taking first non-null for each model col (due to stacking)
        id_cols = ["origin_date","target","horizon_months","y_true"]
        model_cols = [c for c in fc_long.columns if c not in id_cols]
        fc = fc_long.groupby(id_cols, as_index=False).first()
        fc = fc.sort_values(["target","origin_date"]).reset_index(drop=True)

        fc_path = out_dir / f"rnn_forecasts_h{h}.csv"
        fc.to_csv(fc_path, index=False)
        logger.log(f"Wrote {fc_path.name}")

        # Metrics
        mrows = []
        for tgt in sorted(fc["target"].unique()):
            sub = fc[fc["target"] == tgt]
            yt = sub["y_true"].values.astype(float)
            for m in model_cols:
                yh = sub[m].values.astype(float)
                v = np.isfinite(yt) & np.isfinite(yh)
                if v.sum() < 20:
                    continue
                ytv, yhv = yt[v], yh[v]
                rmse = float(np.sqrt(np.mean((yhv - ytv)**2)))
                mae = float(np.mean(np.abs(yhv - ytv)))
                da = directional_accuracy(ytv, yhv)
                tp = turning_point_f1(ytv, yhv)
                mrows.append({"target": tgt, "horizon_months": h, "model": m, "n_obs": int(v.sum()),
                              "rmse": rmse, "mae": mae, "directional_accuracy": da, "turning_point_f1": tp})
        met = pd.DataFrame(mrows)
        met_path = out_dir / f"rnn_metrics_h{h}.csv"
        met.to_csv(met_path, index=False)
        logger.log(f"Wrote {met_path.name}")

        # DM vs VAR
        drows = []
        # join with baseline VAR errors
        merged = base_fc.merge(fc, on=["origin_date","target","horizon_months","y_true"], how="inner")
        for tgt in sorted(merged["target"].unique()):
            sub = merged[merged["target"] == tgt]
            yt = sub["y_true"].values.astype(float)
            e_var = sub["VAR"].values.astype(float) - yt
            for m in model_cols:
                e_m = sub[m].values.astype(float) - yt
                dm, pv = dm_test_sqerr(e_m, e_var)
                drows.append({"target": tgt, "horizon_months": h, "model": m, "benchmark": "VAR", "dm_stat": dm, "p_value": pv})
        dm = pd.DataFrame(drows)
        dm_path = out_dir / f"rnn_dm_vs_var_h{h}.csv"
        dm.to_csv(dm_path, index=False)
        logger.log(f"Wrote {dm_path.name}")

    logger.log("DONE.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run pseudo-OOS RNNs (LSTM/GRU) for selected horizons/models/optimizers.")
    parser.add_argument("--horizons", type=str, default="1,12",
                        help="Comma-separated horizons to run, e.g. '12' or '1,12'.")
    parser.add_argument("--models", type=str, default="LSTM,GRU",
                        help="Comma-separated RNN types to run: LSTM,GRU. Example: 'LSTM'")
    parser.add_argument("--optimizers", type=str, default="Adam,SGD",
                        help="Comma-separated optimizers to run: Adam,SGD. Default: Adam,SGD.")
    parser.add_argument("--retrain-every", type=int, default=None,
                        help="Override retrain frequency in months (e.g. 12 to speed up).")
    args = parser.parse_args()

    cfg = Config()
    # horizons
    hs = [int(x.strip()) for x in args.horizons.split(",") if x.strip()]
    cfg.horizons = tuple(hs)  # type: ignore

    # models
    ms = [x.strip().upper() for x in args.models.split(",") if x.strip()]
    cfg.model_types = tuple(ms)  # type: ignore

    # optimizers
    opts = [x.strip().upper() for x in args.optimizers.split(",") if x.strip()]
    cfg.use_adam = ("ADAM" in opts)
    cfg.use_sgd = ("SGD" in opts)

    if args.retrain_every is not None:
        cfg.retrain_every = int(args.retrain_every)

    main(cfg)
