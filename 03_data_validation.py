"""03_data_validation.py

Purpose
- Run systematic quality checks on the processed monthly panel and create
  diagnostic outputs that document data coverage, missingness, transformations,
  and (optional) stationarity tests.

What the script does
1) Loads the processed monthly panel.
2) Produces descriptive diagnostics:
   - start/end dates and number of observations per series,
   - share and pattern of missing values,
   - basic summary statistics and outlier flags (where implemented).
3) Constructs transformed variants used in forecasting (e.g., YoY growth, changes,
   log differences), while keeping clear naming conventions.
4) Runs unit-root / stationarity diagnostics:
   - Augmented Dickey–Fuller (ADF): tests the null of a unit root.
   - KPSS: tests the null of stationarity.
   The script records p-values (when computable) and a simple “stationary” flag
   based on a joint criterion using both tests.

Outputs
- A set of CSV files in the project’s validation/diagnostics folder, including:
  - per-series descriptive diagnostics (coverage, missingness),
  - stationarity test results for level and transformed series,
  - a dashboard-style summary table that aggregates key flags for quick review.

Interpretation note (for users of the outputs)
- ADF and KPSS answer different null hypotheses; using both helps avoid relying on
  a single test that can be sensitive to sample size, structural breaks, and
  persistence typical of macro time series. The “stationary” flag is therefore
  a diagnostic aid rather than a hard rule that forces a transformation.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib
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

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from statsmodels.tsa.stattools import adfuller, kpss, zivot_andrews
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy.stats import f as f_dist


# =========================================================
# Folder handling
# =========================================================
def ensure_folder_structure(base_data_dir: Path) -> Dict[str, Path]:
    root = base_data_dir / "Data validation"
    paths = {
        "root": root,
        "inputs": root / "01_inputs",
        "tables": root / "02_reports" / "tables",
        "figures": root / "02_reports" / "figures",
        "logs": root / "03_outputs" / "logs",
        "out_model": root / "03_outputs" / "model_ready",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths


# =========================================================
# Utilities
# =========================================================
def safe_to_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def yoy_logdiff(x: pd.Series) -> pd.Series:
    x = safe_to_numeric(x)
    x = x.where(x > 0)
    return 100.0 * (np.log(x) - np.log(x.shift(12)))


def yoy_diff(x: pd.Series) -> pd.Series:
    x = safe_to_numeric(x)
    return x - x.shift(12)


def robust_zscore(x: pd.Series) -> pd.Series:
    x = safe_to_numeric(x)
    v = x.values.astype(float)
    med = np.nanmedian(v)
    mad = np.nanmedian(np.abs(v - med))
    if mad == 0 or np.isnan(mad):
        return pd.Series(np.nan, index=x.index)
    return 0.6745 * (x - med) / mad


def month_dummies(dates: pd.Series) -> pd.DataFrame:
    m = pd.to_datetime(dates).dt.month
    return pd.get_dummies(m, prefix="m", drop_first=True)


def base_name(v: str) -> str:
    return v[:-4] if v.endswith("_yoy") else v


def finite_masked(df: pd.DataFrame, ycol: str) -> Tuple[np.ndarray, pd.Series]:
    y_raw = df[ycol].values.astype(float)
    mask = np.isfinite(y_raw)
    return y_raw[mask], df.loc[mask, "date"].reset_index(drop=True)


# =========================================================
# Categories + transform rules
# =========================================================
CATEGORY_KEYWORDS = [
    ("inflation_prices", ["cpi", "ppi", "pcepi", "deflator", "price", "inflation"]),
    ("real_activity", ["indpro", "ip_", "retail", "sales", "pmi", "orders", "shipments", "inventory", "gdp", "income"]),
    ("labor_market", ["unemployment", "unrate", "payems", "payroll", "employment", "claims", "ahe", "hours", "lfpr", "participation"]),
    ("policy_rates", ["fedfunds", "effr", "sofr", "iorb", "rrp", "rate", "target"]),
    ("yields_curve", ["t10y", "t5y", "t2y", "t3m", "yield", "curve", "term", "spread", "breakeven"]),
    ("financial_conditions", ["sp500", "vix", "credit", "cdx", "hy", "ig", "dollar", "fx", "usd", "swap", "ois"]),
    ("expectations_surveys", ["expect", "survey", "umich", "ref_median_exp", "infl_exp"]),
]

def assign_category(name: str) -> str:
    n = name.lower()
    for cat, kws in CATEGORY_KEYWORDS:
        if any(k in n for k in kws):
            return cat
    return "other"


def classify_rule(name: str) -> str:
    n = name.lower()

    if n.endswith("_yoy") or "yoy" in n or "year_over_year" in n:
        return "KEEP_LEVEL"

    if ("unemployment" in n or "unrate" in n) and ("claims" not in n) and ("insured" not in n):
        return "KEEP_LEVEL"

    if any(k in n for k in ["rate", "yield", "spread", "breakeven", "ffr", "fedfunds", "sofr", "ois", "swap"]):
        return "KEEP_LEVEL"

    if "cpi" in n or "ppi" in n or n.startswith("pcepi") or "deflator" in n:
        return "YOY_LOGDIFF"

    if any(k in n for k in ["indpro", "sales", "income", "payems", "payroll", "employment", "gdp"]):
        return "YOY_LOGDIFF"

    return "KEEP_LEVEL"


def load_variable_dictionary(paths: Dict[str, Path]) -> Optional[pd.DataFrame]:
    f = paths["inputs"] / "variable_dictionary.csv"
    if not f.exists():
        return None
    df = pd.read_csv(f)
    df.columns = [c.strip().lower() for c in df.columns]
    if "var" not in df.columns or "rule" not in df.columns:
        raise ValueError("variable_dictionary.csv must contain columns: var, rule")
    df["var"] = df["var"].astype(str)
    df["rule"] = df["rule"].astype(str).str.upper().str.strip()
    return df


# =========================================================
# Tests
# =========================================================
@dataclass
class UnitRootResult:
    var: str
    form: str
    n: int
    adf_p: Optional[float]
    kpss_p_const: Optional[float]
    stationary_flag: int
    stationary_note: str


@dataclass
class SeasonalityResult:
    var: str
    form: str
    n: int
    f_p_value: Optional[float]
    ljungbox_p_lag12: Optional[float]
    note: str


@dataclass
class BreakResult:
    var: str
    form: str
    n: int
    za_p_value: Optional[float]
    za_break_date: Optional[str]
    za_used_lag: Optional[int]
    note: str


def run_adf(x: pd.Series, maxlag: int) -> Optional[float]:
    x = safe_to_numeric(x).dropna()
    if len(x) < 40 or float(np.nanstd(x.values)) == 0.0:
        return None
    try:
        return float(adfuller(x.values, maxlag=maxlag, regression="c", autolag="AIC")[1])
    except Exception:
        return None


def run_kpss(x: pd.Series) -> Optional[float]:
    x = safe_to_numeric(x).dropna()
    if len(x) < 40 or float(np.nanstd(x.values)) == 0.0:
        return None
    try:
        return float(kpss(x.values, regression="c", nlags="auto")[1])
    except Exception:
        return None


def classify_stationary(adf_p: Optional[float], kpss_p: Optional[float], alpha: float) -> Tuple[int, str]:
    if adf_p is None or kpss_p is None or (not np.isfinite(adf_p)) or (not np.isfinite(kpss_p)):
        return 0, "missing_pvals"
    # Stationary only if both tests agree
    if (adf_p < alpha) and (kpss_p > alpha):
        return 1, "ok"
    return 0, "ok"


def unit_root_suite(var: str, x: pd.Series, form: str, maxlag: int, alpha: float) -> UnitRootResult:
    n = int(safe_to_numeric(x).dropna().shape[0])
    adf_p = run_adf(x, maxlag=maxlag)
    kpss_p = run_kpss(x)
    flag, note = classify_stationary(adf_p, kpss_p, alpha=alpha)
    if n < 40 or float(np.nanstd(safe_to_numeric(x).dropna().values.astype(float))) == 0.0:
        # Explicitly mark why p-values are typically missing
        if note == "missing_pvals":
            note = "too_short_or_constant"
    return UnitRootResult(var, form, n, adf_p, kpss_p, int(flag), note)


def manual_month_f_test_with_trend(dates: pd.Series, y: pd.Series) -> Tuple[Optional[float], str]:
    df = pd.DataFrame({"date": pd.to_datetime(dates), "y": safe_to_numeric(y)}).dropna()
    n = len(df)
    if n < 60:
        return None, "too_short"
    if float(np.nanstd(df["y"].values)) == 0.0:
        return None, "constant"

    Xd = month_dummies(df["date"])
    if Xd.shape[1] == 0:
        return None, "no_dummies"

    yv = df["y"].values.astype(float)
    t = np.arange(n, dtype=float)

    Xr = np.column_stack([np.ones(n, dtype=float), t])
    Xu = np.column_stack([np.ones(n, dtype=float), t, Xd.values])

    try:
        br, *_ = np.linalg.lstsq(Xr, yv, rcond=None)
        bu, *_ = np.linalg.lstsq(Xu, yv, rcond=None)
    except Exception:
        return None, "lstsq_fail"

    resid_r = yv - Xr @ br
    resid_u = yv - Xu @ bu

    rss_r = float(resid_r.T @ resid_r)
    rss_u = float(resid_u.T @ resid_u)

    q = Xu.shape[1] - Xr.shape[1]
    df2 = n - Xu.shape[1]
    if df2 <= 0 or rss_u <= 0:
        return None, "df_or_rss_issue"

    F = ((rss_r - rss_u) / q) / (rss_u / df2)
    if not np.isfinite(F) or F < 0:
        return None, "F_invalid"

    pval = 1.0 - float(f_dist.cdf(F, q, df2))
    return pval, "ok"


def seasonality_suite(base_var: str, dates: pd.Series, series: pd.Series, form: str) -> SeasonalityResult:
    f_p, note = manual_month_f_test_with_trend(dates, series)

    lb_p = None
    try:
        s = safe_to_numeric(series).dropna().values.astype(float)
        s = s[np.isfinite(s)]
        if len(s) >= 60 and float(np.nanstd(s)) > 0.0:
            lb = acorr_ljungbox(s, lags=[12], return_df=True)
            lb_p = float(lb["lb_pvalue"].iloc[0])
    except Exception:
        lb_p = None

    n = int(pd.DataFrame({"y": safe_to_numeric(series)}).dropna().shape[0])
    return SeasonalityResult(base_var, form, n, f_p, lb_p, note)


def _try_zivot(y: np.ndarray, maxlag: int, regression: str, autolag) -> Tuple[Optional[float], Optional[int], Optional[int], str]:
    try:
        res = zivot_andrews(y, maxlag=maxlag, regression=regression, autolag=autolag)
        if not isinstance(res, (tuple, list)):
            return None, None, None, "unexpected_return_type"
        if len(res) == 4:
            _, pval, _, bpidx = res
            usedlag = None
        elif len(res) == 5:
            _, pval, _, usedlag, bpidx = res
            usedlag = int(usedlag) if usedlag is not None else None
        else:
            pval = res[1]
            bpidx = res[-1]
            usedlag = None

        pval = float(pval) if np.isfinite(pval) else None
        bp_i = int(np.asarray(bpidx).reshape(-1)[0])
        return pval, usedlag, bp_i, "ok"
    except Exception as e:
        return None, None, None, f"fail:{type(e).__name__}:{str(e)[:140]}"


def break_suite(colname: str, dates: pd.Series, series: pd.Series, form: str, maxlag: int) -> BreakResult:
    df = pd.DataFrame({"date": pd.to_datetime(dates), "y": safe_to_numeric(series)}).dropna()
    y, finite_dates = finite_masked(df, "y")
    n = int(len(y))

    if n < 80:
        return BreakResult(colname, form, n, None, None, None, "too_short")
    if float(np.nanstd(y)) == 0.0:
        return BreakResult(colname, form, n, None, None, None, "constant")

    attempts = [
        ("c", "AIC", maxlag),
        ("c", "AIC", min(maxlag, max(1, n // 20))),
        ("ct", "AIC", min(maxlag, max(1, n // 20))),
        ("c", None, min(maxlag, max(1, n // 30))),
    ]

    last_note = "not_run"
    for regression, autolag, ml in attempts:
        pval, usedlag, bp_i, note = _try_zivot(y, maxlag=int(ml), regression=regression, autolag=autolag)
        last_note = note
        if note == "ok" and pval is not None and bp_i is not None:
            if 0 <= bp_i < len(finite_dates):
                bp_date = finite_dates.iloc[bp_i].strftime("%Y-%m-%d")
                return BreakResult(colname, form, n, pval, bp_date, usedlag, "ok")
            return BreakResult(colname, form, n, pval, None, usedlag, "break_idx_oob")

    return BreakResult(colname, form, n, None, None, None, last_note)


# =========================================================
# Missingness
# =========================================================
def missingness_profile(dates: pd.Series, x: pd.Series) -> Dict[str, object]:
    s = safe_to_numeric(x)
    d = pd.to_datetime(dates)
    n_total = int(len(s))
    n_missing = int(s.isna().sum())
    overall_pct = float(100.0 * n_missing / n_total) if n_total else np.nan

    if s.notna().sum() == 0:
        return {
            "n": n_total,
            "n_missing": n_missing,
            "overall_missing_pct": overall_pct,
            "within_window_missing_pct": np.nan,
            "first_valid_date": None,
            "last_valid_date": None,
            "late_start_flag": 0,
        }

    first_idx = int(s.first_valid_index())
    last_idx = int(s.last_valid_index())
    first_date = d.iloc[first_idx]
    last_date = d.iloc[last_idx]

    window = s.iloc[first_idx:last_idx + 1]
    w_n = int(len(window))
    w_miss = int(window.isna().sum())
    w_pct = float(100.0 * w_miss / w_n) if w_n else np.nan

    late_start = int((first_date - d.min()).days > 730)

    return {
        "n": n_total,
        "n_missing": n_missing,
        "overall_missing_pct": overall_pct,
        "within_window_missing_pct": w_pct,
        "first_valid_date": first_date.strftime("%Y-%m-%d"),
        "last_valid_date": last_date.strftime("%Y-%m-%d"),
        "late_start_flag": late_start,
    }


# =========================================================
# Minimal QC figures (optional)
# =========================================================
def _savefig(outpath: Path) -> None:
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def plot_missingness_heatmap(panel: pd.DataFrame, outpath: Path, max_vars: int = 60) -> None:
    cols = [c for c in panel.columns if c != "date"][:max_vars]
    m = panel[cols].isna().astype(int).T.values
    plt.figure(figsize=(12, 0.25 * len(cols) + 2))
    plt.imshow(m, aspect="auto", interpolation="nearest")
    plt.yticks(range(len(cols)), cols)
    plt.xticks([])
    plt.title("Missingness heatmap (1=missing)")
    _savefig(outpath)


def plot_missingness_hist(missingness: pd.DataFrame, outpath: Path) -> None:
    plt.figure(figsize=(10, 4))
    plt.hist(missingness["within_window_missing_pct"].dropna().values, bins=30)
    plt.title("Within-window missingness (%)")
    plt.xlabel("Percent")
    plt.ylabel("Count")
    _savefig(outpath)


def plot_outliers_hist(outliers: pd.DataFrame, outpath: Path) -> None:
    plt.figure(figsize=(10, 4))
    plt.hist(outliers["n_outliers_abs_robust_z_gt_5"].values, bins=30)
    plt.title("Outlier counts (|robust z| > 5)")
    plt.xlabel("Count")
    plt.ylabel("Variables")
    _savefig(outpath)


# =========================================================
# Paths
# =========================================================
def default_desktop_data_dir() -> Path:
    # Kept for backwards compatibility; now points to the repo-local Data/ folder.
    return DATA_DIR
def default_input_path() -> Path:
    return default_desktop_data_dir() / "Processed data" / "processed_panel.csv"


# =========================================================
# Dashboard
# =========================================================
def build_dashboard(taxonomy: pd.DataFrame,
                    missingness: pd.DataFrame,
                    outliers: pd.DataFrame,
                    stationarity: pd.DataFrame,
                    seasonality: pd.DataFrame,
                    breaks: pd.DataFrame,
                    alpha: float) -> pd.DataFrame:

    dash = taxonomy.rename(columns={"var": "base_var"}).copy()

    miss = missingness.copy()
    miss["base_var"] = miss["var"].astype(str).apply(base_name)
    miss_agg = miss.groupby("base_var", as_index=False).agg(
        first_valid_date=("first_valid_date", "first"),
        last_valid_date=("last_valid_date", "first"),
        within_window_missing_pct=("within_window_missing_pct", "max"),
        overall_missing_pct=("overall_missing_pct", "max"),
        late_start_flag=("late_start_flag", "max"),
    )

    out = outliers.copy()
    out["base_var"] = out["var"].astype(str).apply(base_name)
    out_agg = out.groupby("base_var", as_index=False).agg(
        n_outliers=("n_outliers_abs_robust_z_gt_5", "max"),
    )

    ur = stationarity.copy()
    ur["base_var"] = ur["var"].astype(str).apply(base_name)
    ur_level = ur[ur["form"] == "level"].groupby("base_var", as_index=False).agg(
        adf_p_level=("adf_p", "min"),
        kpss_p_level=("kpss_p_const", "min"),
    )
    ur_yoy = ur[ur["form"] == "yoy"].groupby("base_var", as_index=False).agg(
        adf_p_yoy=("adf_p", "min"),
        kpss_p_yoy=("kpss_p_const", "min"),
    )

    seas_level = seasonality[seasonality["form"] == "level"].groupby("var", as_index=False).agg(
        seas_f_p_level=("f_p_value", "min"),
        seas_lb12_p_level=("ljungbox_p_lag12", "min"),
        seas_note_level=("note", "first"),
    ).rename(columns={"var": "base_var"})
    seas_yoy = seasonality[seasonality["form"] == "yoy"].groupby("var", as_index=False).agg(
        seas_f_p_yoy=("f_p_value", "min"),
        seas_lb12_p_yoy=("ljungbox_p_lag12", "min"),
        seas_note_yoy=("note", "first"),
    ).rename(columns={"var": "base_var"})

    br = breaks.copy()
    br["base_var"] = br["var"].astype(str).apply(base_name)
    br_agg = br.groupby("base_var", as_index=False).agg(
        za_p_value=("za_p_value", "min"),
        za_break_date=("za_break_date", "first"),
        za_used_lag=("za_used_lag", "first"),
        za_note=("note", "first"),
    )

    dash = dash.merge(miss_agg, on="base_var", how="left")
    dash = dash.merge(out_agg, on="base_var", how="left")
    dash = dash.merge(ur_level, on="base_var", how="left")
    dash = dash.merge(ur_yoy, on="base_var", how="left")
    dash = dash.merge(seas_level, on="base_var", how="left")
    dash = dash.merge(seas_yoy, on="base_var", how="left")
    dash = dash.merge(br_agg, on="base_var", how="left")

    dash["seasonality_mean_level_flag"] = ((dash["seas_note_level"] == "ok") & (dash["seas_f_p_level"] < alpha)).astype(int)
    dash["seasonality_mean_yoy_flag"] = ((dash["seas_note_yoy"] == "ok") & (dash["seas_f_p_yoy"] < alpha)).astype(int)
    dash["break_flag"] = ((dash["za_note"] == "ok") & (dash["za_p_value"] < alpha)).astype(int)

    # NEW: stationarity flags from aggregated p-values
    dash["stationary_level_flag"] = ((dash["adf_p_level"] < alpha) & (dash["kpss_p_level"] > alpha)).astype(int)
    dash["stationary_yoy_flag"] = ((dash["adf_p_yoy"] < alpha) & (dash["kpss_p_yoy"] > alpha)).astype(int)

    return dash


# =========================================================
# Main
# =========================================================
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=str, default=str(default_input_path()))
    p.add_argument("--data_dir", type=str, default=str(default_desktop_data_dir()))
    p.add_argument("--fast", action="store_true")
    p.add_argument("--maxlag", type=int, default=6)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--no_figures", action="store_true")
    args = p.parse_args()

    data_dir = Path(args.data_dir).expanduser()
    paths = ensure_folder_structure(data_dir)

    log_file = paths["logs"] / "run_log.txt"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.FileHandler(log_file, mode="w"), logging.StreamHandler()],
    )

    input_path = Path(args.input).expanduser()
    if not input_path.exists():
        raise FileNotFoundError(f"Could not find: {input_path}")

    panel = pd.read_csv(input_path)
    if "date" not in panel.columns:
        raise ValueError("Input must include a 'date' column.")
    panel["date"] = pd.to_datetime(panel["date"], errors="coerce")
    panel = panel.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    numeric_cols = [c for c in panel.columns if c != "date"]
    for c in numeric_cols:
        panel[c] = safe_to_numeric(panel[c])

    # Overrides
    var_dict = load_variable_dictionary(paths)
    overrides: Dict[str, str] = {}
    if var_dict is not None:
        overrides = dict(zip(var_dict["var"], var_dict["rule"]))
        logging.info("Loaded variable_dictionary.csv overrides (%d rows).", len(overrides))

    # Build validated dataset + taxonomy
    taxonomy_rows = []
    validated = pd.DataFrame({"date": panel["date"].copy()})

    for c in numeric_cols:
        category = assign_category(c)
        rule = overrides.get(c, classify_rule(c))
        taxonomy_rows.append({"var": c, "category": category, "rule": rule})

        if rule == "KEEP_LEVEL":
            validated[c] = panel[c]
        elif rule == "YOY_LOGDIFF":
            validated[f"{c}_yoy"] = yoy_logdiff(panel[c])
        elif rule == "YOY_DIFF":
            validated[f"{c}_yoy"] = yoy_diff(panel[c])
        else:
            validated[c] = panel[c]

    taxonomy = pd.DataFrame(taxonomy_rows)
    taxonomy.to_csv(paths["tables"] / "variable_taxonomy_and_transforms.csv", index=False)

    # Missingness
    miss_rows = []
    for c in numeric_cols:
        prof = missingness_profile(panel["date"], panel[c])
        prof["var"] = c
        miss_rows.append(prof)

        yoy_name = f"{c}_yoy"
        if yoy_name in validated.columns:
            prof2 = missingness_profile(validated["date"], validated[yoy_name])
            prof2["var"] = yoy_name
            miss_rows.append(prof2)

    missingness = pd.DataFrame(miss_rows)
    missingness.to_csv(paths["tables"] / "missingness_summary.csv", index=False)

    # Outliers
    out_rows = []
    for c in [x for x in validated.columns if x != "date"]:
        out_rows.append({
            "var": c,
            "n_non_missing": int(validated[c].notna().sum()),
            "n_outliers_abs_robust_z_gt_5": int((np.abs(robust_zscore(validated[c])) > 5).sum()),
        })
    outliers = pd.DataFrame(out_rows).sort_values("n_outliers_abs_robust_z_gt_5", ascending=False)
    outliers.to_csv(paths["tables"] / "outlier_summary.csv", index=False)

    # Stationarity (+ flags)
    ur_rows = []
    for c in [x for x in validated.columns if x != "date"]:
        form = "yoy" if c.endswith("_yoy") else "level"
        ur_rows.append(unit_root_suite(c, validated[c], form, maxlag=args.maxlag, alpha=args.alpha).__dict__)
    stationarity = pd.DataFrame(ur_rows)
    stationarity.to_csv(paths["tables"] / "stationarity_tests.csv", index=False)

    # Seasonality
    seas_rows = []
    for c in numeric_cols:
        seas_rows.append(seasonality_suite(c, panel["date"], panel[c], "level").__dict__)
        yoy_name = f"{c}_yoy"
        if yoy_name in validated.columns:
            seas_rows.append(seasonality_suite(c, panel["date"], validated[yoy_name], "yoy").__dict__)
    seasonality = pd.DataFrame(seas_rows)
    seasonality.to_csv(paths["tables"] / "seasonality_tests.csv", index=False)

    # Structural breaks
    candidates = [c for c in validated.columns if c != "date"]
    eligible = []
    for c in candidates:
        s = safe_to_numeric(validated[c]).dropna().values.astype(float)
        s = s[np.isfinite(s)]
        if len(s) >= 80 and float(np.nanstd(s)) > 0.0:
            eligible.append(c)
    if args.fast:
        eligible = eligible[:60]

    br_rows = []
    for c in eligible:
        form = "yoy" if c.endswith("_yoy") else "level"
        br_rows.append(break_suite(c, validated["date"], validated[c], form, maxlag=args.maxlag).__dict__)
    breaks = pd.DataFrame(br_rows)
    breaks.to_csv(paths["tables"] / "structural_break_tests.csv", index=False)

    # Dashboard (includes stationarity flags)
    dashboard = build_dashboard(taxonomy, missingness, outliers, stationarity, seasonality, breaks, alpha=args.alpha)
    dashboard.to_csv(paths["tables"] / "dashboard_summary.csv", index=False)

    # Results summary (tight)
    cols = [
        "base_var","category","rule",
        "within_window_missing_pct","late_start_flag",
        "n_outliers",
        "adf_p_level","kpss_p_level","stationary_level_flag",
        "adf_p_yoy","kpss_p_yoy","stationary_yoy_flag",
        "seas_f_p_level","seas_note_level","seas_f_p_yoy","seas_note_yoy",
        "za_p_value","za_break_date","za_note",
        "seasonality_mean_level_flag","seasonality_mean_yoy_flag","break_flag"
    ]
    cols = [c for c in cols if c in dashboard.columns]
    dashboard[cols].to_csv(paths["tables"] / "results_summary.csv", index=False)

    # Model-ready dataset
    out_csv = paths["out_model"] / "model_ready_base.csv"
    validated.to_csv(out_csv, index=False)

    # Minimal figures (optional)
    if not args.no_figures:
        plot_missingness_heatmap(panel, paths["figures"] / "missingness_heatmap.png")
        plot_missingness_hist(missingness, paths["figures"] / "missingness_hist.png")
        plot_outliers_hist(outliers, paths["figures"] / "outliers_hist.png")

    logging.info("Wrote tables: %s", paths["tables"].resolve())
    logging.info("Done.")


if __name__ == "__main__":
    main()
