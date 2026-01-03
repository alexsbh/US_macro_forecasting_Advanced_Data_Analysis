"""04a_pca_factors.py

Purpose
- Create a small set of latent “common factor” predictors using Principal
  Component Analysis (PCA) applied to the monthly macro panel.

Why PCA factors are useful
- When many predictors are correlated, PCA summarizes their shared variation into
  a few orthogonal components.
- These components can be used as additional features in forecasting models to
  capture broad macro/financial conditions with a limited number of regressors.

Methodology implemented
1) Load the processed monthly panel and select the candidate predictor set.
2) Fit preprocessing steps on a training subsample only:
   - imputation for missing values,
   - standardization to comparable units,
   - PCA to obtain the rotation matrix and component scores.
   Fitting on an initial subsample avoids using future information when building
   features for pseudo out-of-sample forecasting.
3) Apply the fitted preprocessing + PCA rotation to the full sample to obtain
   component scores for every month.
4) Write the resulting principal component series (PC1, PC2, …) to disk.

Inputs
- Processed monthly panel produced earlier in the pipeline.

Outputs
- A CSV (or equivalent) containing principal component time series saved in the
  project’s results/data folder, ready to be merged into model design matrices.
"""

from __future__ import annotations

import os
import time
import warnings
from dataclasses import dataclass
from typing import List, Tuple

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


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

@dataclass
class PCAConfig:
    date_col: str = "date"

    # Load the processed monthly panel used as the PCA input universe.
    data_path: str = ""
    out_dir: str = ""

    # Fit PCA preprocessing on an initial subsample only to avoid using future information.
    pca_train_end: str = "1999-12-01"  # fit PCA on <= this date

    # Maximum number of principal components to compute and export.
    n_components_max: int = 8

    # Exclude non-predictor columns (dates, targets, identifiers) from the PCA input matrix.
    exclude_cols: Tuple[str, ...] = (
        "cpi_headline_yoy",
        "cpi_core_yoy",
        "indpro_total_yoy",
        "unrate_u3",
        "t10y2y",
        "effr",
    )


class Logger:
    def __init__(self, out_dir: str):
        os.makedirs(out_dir, exist_ok=True)
        self.path = os.path.join(out_dir, "run_log_pca.txt")

    def log(self, msg: str):
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line, flush=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def _default_paths() -> Tuple[str, str]:
    base = str(PROJECT_ROOT)
    data_path = os.path.join(
        base, "Data", "Data validation", "03_outputs", "model_ready", "model_ready_base.csv"
    )
    out_dir = os.path.join(base, "Data", "Pseudo OOS Forecasting Results")
    return data_path, out_dir


def select_numeric_pca_cols(df: pd.DataFrame, date_col: str, exclude: List[str]) -> List[str]:
    cols: List[str] = []
    for c in df.columns:
        if c == date_col or c in exclude:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols


def run_pca(cfg: PCAConfig) -> None:
    warnings.filterwarnings("ignore")
    if not cfg.data_path or not cfg.out_dir:
        cfg.data_path, cfg.out_dir = _default_paths()

    lg = Logger(cfg.out_dir)
    lg.log("Starting PCA factor build")

    df = pd.read_csv(cfg.data_path)
    if cfg.date_col not in df.columns:
        raise ValueError(f"Missing date column: {cfg.date_col}")
    df[cfg.date_col] = pd.to_datetime(df[cfg.date_col])
    df = df.sort_values(cfg.date_col).set_index(cfg.date_col)

    lg.log(
        f"Data range: {df.index.min().date()} .. {df.index.max().date()} | rows={len(df)} cols={df.shape[1]}"
    )

    pca_cols = select_numeric_pca_cols(df, cfg.date_col, list(cfg.exclude_cols))
    if len(pca_cols) < 5:
        raise ValueError(f"Not enough numeric columns for PCA after exclusions. Have {len(pca_cols)}.")

    train_end = pd.to_datetime(cfg.pca_train_end)
    train_mask = df.index <= train_end

    X_train = df.loc[train_mask, pca_cols].copy()
    X_all = df.loc[:, pca_cols].copy()

    # Remove columns that are entirely missing in the training subsample (PCA cannot be fit on them).
    all_nan = X_train.isna().all(axis=0)
    if bool(all_nan.any()):
        keep_cols = list(X_train.columns[~all_nan])
        lg.log(f"Dropping {int(all_nan.sum())} all-NaN cols in PCA train window.")
        X_train = X_train[keep_cols].copy()
        X_all = X_all[keep_cols].copy()
        pca_cols = keep_cols

    ncomp = min(cfg.n_components_max, len(pca_cols))
    lg.log(f"Fitting PCA with n_components={ncomp} (max requested={cfg.n_components_max})")

    imp = SimpleImputer(strategy="mean")
    sc = StandardScaler()
    X_train_std = sc.fit_transform(imp.fit_transform(X_train))

    pca = PCA(n_components=ncomp, random_state=42).fit(X_train_std)

    X_all_std = sc.transform(imp.transform(X_all))
    PCs = pca.transform(X_all_std)

    pc_names = [f"PC{i+1}" for i in range(ncomp)]
    factors = pd.DataFrame(PCs, index=df.index, columns=pc_names)

    # Save PCA loadings: components_ has shape (n_components, n_features).
    loadings = pd.DataFrame(pca.components_.T, index=pca_cols, columns=pc_names)

    explained = pd.DataFrame(
        {
            "pc": pc_names,
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "explained_variance": pca.explained_variance_,
        }
    )
    explained["cum_explained_variance_ratio"] = explained["explained_variance_ratio"].cumsum()

    out_factors = os.path.join(cfg.out_dir, "pca_factors.csv")
    out_loadings = os.path.join(cfg.out_dir, "pca_loadings.csv")
    out_expl = os.path.join(cfg.out_dir, "pca_explained_variance.csv")

    factors.reset_index().rename(columns={cfg.date_col: "date"}).to_csv(out_factors, index=False)
    loadings.reset_index().rename(columns={"index": "variable"}).to_csv(out_loadings, index=False)
    explained.to_csv(out_expl, index=False)

    lg.log(f"Saved: {out_factors}")
    lg.log(f"Saved: {out_loadings}")
    lg.log(f"Saved: {out_expl}")
    lg.log("Done.")


if __name__ == "__main__":
    run_pca(PCAConfig())
