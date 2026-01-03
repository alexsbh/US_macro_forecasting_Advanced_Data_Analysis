"""01_data_scraper.py

Purpose
- Download a set of macro/financial time series and save them in a consistent,
  project-ready format.
- Two sources are supported in this script:
  1) FRED (via the official FRED API) for most public U.S. macro series.
  2) Refinitiv (via Eikon/Refinitiv Data API) for market series that are not on FRED.

What the script does
1) Loops over a curated list of FRED series identifiers.
2) For each series:
   - requests the full observation history from the FRED API,
   - parses the JSON payload into a tidy DataFrame,
   - converts the time index to monthly frequency by taking the last available
     observation within each calendar month (dates are set to month-start),
   - writes the monthly series to disk.
3) Pulls selected Refinitiv series (based on a mapping defined in the script),
   standardizes their date/value columns, and writes them to disk.
4) Creates/updates an audit log so every downloaded series has provenance and
   basic diagnostics (availability window, number of observations, status).

Inputs
- API keys loaded from environment variables or local configuration (FRED key;
  Refinitiv/Eikon key if Refinitiv series are used).
- The list of FRED series codes and the Refinitiv mapping defined below.

Outputs
- One CSV per series in the project’s raw-data directory (monthly, tidy format).
- An audit CSV that records download status and basic metadata for reproducibility.
"""

from __future__ import annotations

import os
import time
import random
import requests
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional, Any, Tuple


# =============================================================================
# Config
# =============================================================================

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

FRED_OBS_URL = "https://api.stlouisfed.org/fred/series/observations"
DEFAULT_OUTDIR = DATA_DIR / "Raw data"
DEFAULT_REFINITIV_FILE = PROJECT_ROOT / "Combined_Refinitiv_data.xlsx"

OBS_START = "1900-01-01"
OBS_END   = "2100-01-01"

# =============================================================================
# Series list (variable names mapped to FRED codes)
# =============================================================================
SERIES: List[Dict[str, str]] = [
    {"name": "cpi_headline", "fred_code": "CPIAUCSL"},
    {"name": "cpi_core", "fred_code": "CPILFESL"},
    {"name": "cpi_services", "fred_code": "CUSR0000SA0L2"},
    {"name": "cpi_shelter", "fred_code": "CUSR0000SAH1"},
    {"name": "cpi_rent", "fred_code": "CUSR0000SEHA"},
    {"name": "cpi_oer", "fred_code": "CUSR0000SEHC"},
    {"name": "cpi_food_home", "fred_code": "CUSR0000SAF11"},
    {"name": "cpi_energy_services", "fred_code": "CUSR0000SEHF"},
    {"name": "cpi_transport", "fred_code": "CUSR0000SAT1"},
    {"name": "cpi_med_services", "fred_code": "CUSR0000SAM2"},
    {"name": "cpi_med_commod", "fred_code": "CUSR0000SAM1"},
    {"name": "cpi_comm_less_fe", "fred_code": "CUSR0000SACL1E"},
    {"name": "pcepi_headline", "fred_code": "PCEPI"},
    {"name": "pcepi_core", "fred_code": "PCEPILFE"},
    {"name": "gdp_deflator", "fred_code": "GDPDEF"},
    # (keeping your original list; note you have ppi_all_commodities and ppi_all_manufacturing with same code)
    {"name": "ppi_all_commodities", "fred_code": "PCUOMFGOMFG"},
    {"name": "ppi_all_manufacturing", "fred_code": "PCUOMFGOMFG"},
    {"name": "ppi_mining", "fred_code": "PCUOMINOMIN"},
    {"name": "breakeven_5y", "fred_code": "T5YIE"},
    {"name": "breakeven_10y", "fred_code": "T10YIE"},
    {"name": "infl_5y5y_fwd", "fred_code": "T5YIFR"},
    {"name": "umich_infl_1y", "fred_code": "MICH"},
    {"name": "expinf_5y_clevelandfed", "fred_code": "EXPINF5YR"},
    {"name": "indpro_total", "fred_code": "INDPRO"},
    {"name": "indpro_manufacturing", "fred_code": "IPMAN"},
    {"name": "indpro_durables", "fred_code": "IPDMAT"},
    {"name": "indpro_nondurables", "fred_code": "IPNMAT"},
    {"name": "caputil_total", "fred_code": "TCU"},
    {"name": "caputil_manufacturing", "fred_code": "CUMFNS"},
    {"name": "retail_sales_nominal", "fred_code": "RSAFS"},
    {"name": "retail_sales_real", "fred_code": "RRSFS"},
    {"name": "gdp_nominal", "fred_code": "GDP"},
    {"name": "gdp_real", "fred_code": "GDPC1"},
    {"name": "exports_real", "fred_code": "EXPGSC1"},
    {"name": "imports_real", "fred_code": "IMPGSC1"},
    {"name": "durables_new_orders", "fred_code": "DGORDER"},
    {"name": "durables_shipments_value", "fred_code": "AMDMVS"},
    {"name": "durables_unfilled_orders", "fred_code": "AMDMUO"},
    {"name": "manufacturers_inventories", "fred_code": "MNFCTRMPCIMSA"},
    {"name": "business_inventories", "fred_code": "BUSINV"},
    {"name": "retail_inventories", "fred_code": "RETAILIMSA"},
    {"name": "wholesale_inventories", "fred_code": "WHLSLRIMSA"},
    {"name": "lei", "fred_code": "USSLIND"},
    {"name": "coincident_index", "fred_code": "USPHCI"},
    {"name": "cf_nai", "fred_code": "CFNAI"},
    {"name": "oecd_consumer_conf_cli", "fred_code": "CSCICP03USM665S"},
    {"name": "oecd_business_conf_cli", "fred_code": "BSCICP03USM665S"},
    {"name": "oecd_mfg_business_conf", "fred_code": "BSCICP02USM460S"},
    {"name": "housing_starts", "fred_code": "HOUST"},
    {"name": "housing_starts_sf", "fred_code": "HOUST1F"},
    {"name": "housing_starts_mf5", "fred_code": "HOUST5F"},
    {"name": "permits", "fred_code": "PERMIT"},
    {"name": "permits_sf", "fred_code": "PERMIT1"},
    {"name": "new_home_sales", "fred_code": "HSN1F"},
    {"name": "new_home_price_median", "fred_code": "MSPNHSUS"},
    {"name": "new_home_months_supply", "fred_code": "MSACSR"},
    {"name": "case_shiller", "fred_code": "CSUSHPINSA"},
    {"name": "fhfa_hpi_all_transactions", "fred_code": "USSTHPI"},
    {"name": "mortgage_30y", "fred_code": "MORTGAGE30US"},
    {"name": "mortgage_15y", "fred_code": "MORTGAGE15US"},
    {"name": "unrate_u3", "fred_code": "UNRATE"},
    {"name": "unrate_u6", "fred_code": "U6RATE"},
    {"name": "employment_level", "fred_code": "CE16OV"},
    {"name": "unemployment_level", "fred_code": "UNEMPLOY"},
    {"name": "labor_force", "fred_code": "CLF16OV"},
    {"name": "participation", "fred_code": "CIVPART"},
    {"name": "emp_pop_ratio", "fred_code": "EMRATIO"},
    {"name": "payrolls_total", "fred_code": "PAYEMS"},
    {"name": "payrolls_private", "fred_code": "USPRIV"},
    {"name": "payrolls_manufacturing", "fred_code": "MANEMP"},
    {"name": "payrolls_construction", "fred_code": "USCONS"},
    {"name": "payrolls_trade_transport_util", "fred_code": "USTPU"},
    {"name": "payrolls_retail_trade", "fred_code": "USTRADE"},
    {"name": "payrolls_leisure_hosp", "fred_code": "USLAH"},
    {"name": "payrolls_govt", "fred_code": "USGOVT"},
    {"name": "ahe_total", "fred_code": "CES0500000003"},
    {"name": "avg_weekly_hours_mfg", "fred_code": "AWHAEMAN"},
    {"name": "avg_weekly_earnings_mfg", "fred_code": "AWHMAN"},
    {"name": "initial_claims", "fred_code": "ICSA"},
    {"name": "continuing_claims", "fred_code": "CCSA"},
    {"name": "jolts_openings", "fred_code": "JTSJOL"},
    {"name": "jolts_hires", "fred_code": "JTSHIL"},
    {"name": "jolts_quits", "fred_code": "JTSQUL"},
    {"name": "jolts_layoffs", "fred_code": "JTSLDL"},
    {"name": "sloos_tightening_ci_large_mid", "fred_code": "DRTSCILM"},
    {"name": "effr", "fred_code": "EFFR"},
    {"name": "sofr", "fred_code": "SOFR"},
    {"name": "fed_funds_target", "fred_code": "FEDFUNDS"},
    {"name": "tbill_3m", "fred_code": "DTB3"},
    {"name": "ust_2y", "fred_code": "DGS2"},
    {"name": "ust_5y", "fred_code": "DGS5"},
    {"name": "ust_10y", "fred_code": "DGS10"},
    {"name": "ust_30y", "fred_code": "DGS30"},
    {"name": "real_5y", "fred_code": "DFII5"},
    {"name": "real_10y", "fred_code": "DFII10"},
    {"name": "stlfsi", "fred_code": "STLFSI4"},
    {"name": "nfci", "fred_code": "NFCI"},
    {"name": "anfci", "fred_code": "ANFCI"},
    {"name": "us_epu_index", "fred_code": "USEPUINDXD"},
    {"name": "moody_aaa", "fred_code": "AAA"},
    {"name": "moody_baa", "fred_code": "BAA"},
    {"name": "aaa_10y_spread", "fred_code": "AAA10Y"},
    {"name": "baa_10y_spread", "fred_code": "BAA10Y"},
    {"name": "ig_oas", "fred_code": "BAMLC0A0CM"},
    {"name": "hy_oas", "fred_code": "BAMLH0A0HYM2"},
    {"name": "consumer_credit_total", "fred_code": "TOTALSL"},
    {"name": "consumer_credit_revolving", "fred_code": "REVOLSL"},
    {"name": "consumer_credit_nonrevolving", "fred_code": "NONREVSL"},
    {"name": "total_bank_credit", "fred_code": "TOTBKCR"},
    {"name": "commercial_industrial_loans", "fred_code": "BUSLOANS"},
    {"name": "real_estate_loans", "fred_code": "REALLN"},
    {"name": "delinq_consumer_loans", "fred_code": "DRALACBN"},
    {"name": "delinq_credit_cards", "fred_code": "DRCCLACBS"},
    {"name": "chargeoff_credit_cards", "fred_code": "CORCCACBS"},
    {"name": "m1", "fred_code": "M1SL"},
    {"name": "m2", "fred_code": "M2SL"},
    {"name": "monetary_base", "fred_code": "BOGMBASE"},
    {"name": "m2_velocity", "fred_code": "M2V"},
    {"name": "m1_velocity", "fred_code": "M1V"},
    {"name": "real_m2", "fred_code": "M2REAL"},
    {"name": "vix", "fred_code": "VIXCLS"},
    {"name": "dollar_broad", "fred_code": "DTWEXBGS"},
    {"name": "dollar_major", "fred_code": "DTWEXM"},
    {"name": "usd_eur", "fred_code": "DEXUSEU"},
    {"name": "usd_jpy", "fred_code": "DEXJPUS"},
    {"name": "usd_gbp", "fred_code": "DEXUSUK"},
    {"name": "wti", "fred_code": "DCOILWTICO"},
    {"name": "gasoline", "fred_code": "GASREGW"},
    {"name": "natgas", "fred_code": "DHHNGSP"},
    {"name": "copper", "fred_code": "PCOPPUSDM"},
    {"name": "commodity_index_proxy", "fred_code": "PALLFNFINDEXQ"},
    {"name": "t10y2y", "fred_code": "T10Y2Y"},
    {"name": "t10y3mm", "fred_code": "T10Y3MM"},
    {"name": "t5yff", "fred_code": "T5YFF"},
    {"name": "t1yff", "fred_code": "T1YFF"},
    {"name": "cpff", "fred_code": "CPFF"},
    {"name": "tb3smffm", "fred_code": "TB3SMFFM"},
    {"name": "tb6smffm", "fred_code": "TB6SMFFM"},
    {"name": "aaaff", "fred_code": "AAAFF"},
    {"name": "baaff", "fred_code": "BAAFF"},
]

# =============================================================================
# Refinitiv mapping (variable names mapped to Refinitiv instruments/fields)
# =============================================================================
REFINITIV_RENAME_MAP = {
    "SP500": "ref_sp500",
    "AAII Bull Bear Spread": "ref_aaii_bull_bear_spread",
    "CDX NA HY": "ref_cdx_na_hy",
    "CDX NA IG": "ref_cdx_na_ig",
    "Conference Board Consumer Confidence Index": "ref_cb_consumer_confidence",
    "Gold": "ref_gold",
    "Mortgage Applications": "ref_mortgage_applications",
    "Median one-year ahead expected inflation rate": "ref_median_exp_infl_1y",
    "Median three-year ahead expected inflation rate": "ref_median_exp_infl_3y",
    "Russell 2000": "ref_russell_2000",
    "US OIS 3M": "ref_us_ois_3m",
    "USD DXY Index": "ref_usd_dxy_index",
    "World Trade Index": "ref_world_trade_index",
    "ACM Treasury Term Premium 10Y": "ref_acm_term_premium_10y",
    "Export Price Index: All Commodities": "ref_export_price_index",
    "Import Price Index: All Commodities": "ref_import_price_index",
    "University of Michigan: Consumer Sentiment": "ref_unmich",
}

# =============================================================================
# Audit record structure used to log download status and provenance
# =============================================================================
@dataclass
class AuditRow:
    name: str
    fred_code: str
    fred_frequency: Optional[str]
    fred_frequency_short: Optional[str]
    harmonization: str
    n_obs: int
    start: Optional[str]
    end: Optional[str]
    status: str
    notes: str




# =============================================================================
# HTTP helper: request JSON with capped retries and short backoff (handles transient failures).
# =============================================================================

def request_json_retry(
    session: requests.Session,
    url: str,
    params: Dict[str, str],
    timeout: float,
    max_retries: int = 6,
    backoff_base: float = 0.8,
    backoff_cap: float = 2.0,
    jitter: float = 0.25,
) -> Tuple[Optional[dict], Optional[str], Optional[int]]:
    """Request a JSON endpoint with a small, controlled retry loop.

Rules implemented here
- If the request succeeds (HTTP 200), return the decoded JSON immediately.
- If the server is temporarily unavailable or rate-limited (HTTP 429 or 5xx),
  wait briefly (exponential backoff with a cap + small random jitter) and retry.
- For other HTTP errors (e.g., 400/401/403/404), do not retry: return an error
  message so the caller can record a clean failure in the audit log.

Returns
- (json_dict, error_message, http_status)
  where json_dict is None if the request did not succeed.
"""
    last_status: Optional[int] = None

    for attempt in range(1, max_retries + 1):
        try:
            resp = session.get(url, params=params, timeout=timeout)
            last_status = resp.status_code

            if resp.status_code == 200:
                return resp.json(), None, 200

            if resp.status_code == 429 or resp.status_code in (500, 502, 503, 504):
                wait = min(backoff_cap, backoff_base * (2 ** (attempt - 1)))
                wait = wait + random.random() * jitter
                time.sleep(wait)
                continue

            return None, f"HTTP {resp.status_code}: {resp.text[:250]}", resp.status_code

        except requests.RequestException as e:
            wait = min(backoff_cap, backoff_base * (2 ** (attempt - 1)))
            wait = wait + random.random() * jitter
            time.sleep(wait)
            last_status = None

    if last_status == 429:
        return None, "HTTP 429: rate limited (rotate key / wait / reduce concurrent use).", 429

    return None, f"Failed after {max_retries} attempts.", last_status


def to_monthly_last_obs(
    df: pd.DataFrame,
    date_col: str = "date",
) -> pd.DataFrame:
    """Convert a time series to monthly frequency by taking the last observation in each month.

Why "last observation"?
- Many high-frequency series (daily/weekly) need a deterministic monthly mapping.
- Taking the last available value in each calendar month is a standard way to
  represent the month-end level while preserving information from the most recent
  observation.

Implementation details
- Creates a monthly period from the date column.
- Groups by month and keeps the last row.
- Sets the output date to the month start for consistent dating across series.
"""
    if df is None or df.empty:
        return df

    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    out = out.dropna(subset=[date_col]).sort_values(date_col)

    out["_month"] = out[date_col].dt.to_period("M")
    out = out.groupby("_month", as_index=False).last()
    out[date_col] = out["_month"].dt.to_timestamp(how="start")
    out = out.drop(columns=["_month"])
    return out



def fetch_observations(
    session: requests.Session,
    fred_code: str,
    api_key: str,
    timeout: float,
) -> Tuple[pd.DataFrame, Optional[str]]:
    params = {
        "api_key": api_key,
        "file_type": "json",
        "series_id": fred_code,
        "observation_start": OBS_START,
        "observation_end": OBS_END,
    }
    js, err, _ = request_json_retry(session, FRED_OBS_URL, params, timeout=timeout)

    if err is not None or js is None:
        return pd.DataFrame(columns=["date", "value"]), err or "No JSON"

    obs = js.get("observations", [])
    df = pd.DataFrame(obs)
    if df.empty:
        return pd.DataFrame(columns=["date", "value"]), None

    df = df[["date", "value"]].copy()
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["date"])
    return df, None


# =============================================================================
# Refinitiv functions (UNCHANGED)
# =============================================================================
def load_refinitiv_monthly(ref_path: Path, monthly_index: pd.DatetimeIndex) -> pd.DataFrame:
    if not ref_path.exists():
        raise FileNotFoundError(f"Refinitiv file not found: {ref_path}")

    df = pd.read_excel(ref_path)
    if "Date" not in df.columns:
        raise ValueError("Refinitiv Excel must contain a 'Date' column.")

    missing = [c for c in REFINITIV_RENAME_MAP.keys() if c not in df.columns]
    if missing:
        raise ValueError(f"Refinitiv Excel missing expected columns: {missing}")

    df = df[["Date"] + list(REFINITIV_RENAME_MAP.keys())].copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

    df = df.rename(columns={"Date": "date", **REFINITIV_RENAME_MAP})
    for c in df.columns:
        if c != "date":
            df[c] = pd.to_numeric(df[c], errors="coerce")

    m = to_monthly_last_obs(df, date_col="date")
    # Align month-end -> month-start index (explicit)
    # Ensure we have a DatetimeIndex before converting to monthly periods
    if not isinstance(m.index, pd.DatetimeIndex):
        if "date" in m.columns:
            m["date"] = pd.to_datetime(m["date"], errors="coerce")
            m = m.dropna(subset=["date"]).set_index("date")
        else:
            # Fall back: try to interpret the first column as dates
            first_col = m.columns[0]
            m[first_col] = pd.to_datetime(m[first_col], errors="coerce")
            m = m.dropna(subset=[first_col]).set_index(first_col)
    m.index = pd.to_datetime(m.index, errors="coerce")
    m = m[~m.index.isna()]
    m.index = m.index.to_period("M").to_timestamp(how="start")

    return m.reindex(monthly_index).reset_index().rename(columns={"index": "date"})


def append_refinitiv_to_audit(ref_m: pd.DataFrame, audit_rows: List[AuditRow]) -> None:
    """Append Refinitiv series entries to the same audit file used for FRED.

This keeps a single provenance table across data sources so downstream steps
(panel construction, modeling, reporting) can rely on one place to check:
- which series were included,
- whether a download succeeded,
- the covered date range and the number of observations.
"""
    if "date" not in ref_m.columns:
        raise ValueError("ref_m must include a 'date' column")

    dates = pd.to_datetime(ref_m["date"], errors="coerce")
    for original_name, renamed in REFINITIV_RENAME_MAP.items():
        if renamed not in ref_m.columns:
            audit_rows.append(
                AuditRow(
                    name=renamed,
                    fred_code="REFINITIV",
                    fred_frequency="Monthly",
                    fred_frequency_short="M",
                    harmonization="REFINITIV_MONTHLY",
                    n_obs=0,
                    start=None,
                    end=None,
                    status="REFINITIV_MISSING_COL",
                    notes=f"Expected '{renamed}' after rename; original column='{original_name}'",
                )
            )
            continue

        s = pd.to_numeric(ref_m[renamed], errors="coerce")
        nonmiss = s.dropna()
        if nonmiss.empty:
            audit_rows.append(
                AuditRow(
                    name=renamed,
                    fred_code="REFINITIV",
                    fred_frequency="Monthly",
                    fred_frequency_short="M",
                    harmonization="REFINITIV_MONTHLY",
                    n_obs=0,
                    start=None,
                    end=None,
                    status="REFINITIV_ALL_NA",
                    notes=f"All values NA; original column='{original_name}'",
                )
            )
            continue

        start_date = dates.loc[nonmiss.index.min()]
        end_date = dates.loc[nonmiss.index.max()]
        audit_rows.append(
            AuditRow(
                name=renamed,
                fred_code="REFINITIV",
                fred_frequency="Monthly",
                fred_frequency_short="M",
                harmonization="REFINITIV_MONTHLY",
                n_obs=int(nonmiss.shape[0]),
                start=str(start_date.date()) if pd.notna(start_date) else None,
                end=str(end_date.date()) if pd.notna(end_date) else None,
                status="REFINITIV_OK",
                notes=f"original column='{original_name}'",
            )
        )





# =============================================================================
# Main
# =============================================================================

def main():
    api_key = os.environ.get("FRED_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("ERROR: FRED_API_KEY is not set. Example: export FRED_API_KEY='...'\n")

    outdir = DEFAULT_OUTDIR
    outdir.mkdir(parents=True, exist_ok=True)

    ref_path = DEFAULT_REFINITIV_FILE
    if not ref_path.exists():
        raise SystemExit(f"ERROR: Refinitiv file not found: {ref_path.resolve()}")

    session = requests.Session()

    audit_rows: List[AuditRow] = []
    series_monthly: Dict[str, pd.Series] = {}

    # --- FRED ---
    for spec in SERIES:
        name = spec["name"]
        fred_code = spec["fred_code"]

        df_obs, err = fetch_observations(session, fred_code, api_key, timeout=20.0)
        if err is not None:
            audit_rows.append(AuditRow(
                name=name, fred_code=fred_code,
                fred_frequency="Unknown", fred_frequency_short="?",
                harmonization="FRED_NATIVE",
                status="FRED_ERROR", n_obs=0, start=None, end=None, notes=err
            ))
            continue

        df_m = to_monthly_last_obs(df_obs)
        if df_m.empty:
            audit_rows.append(AuditRow(
                name=name, fred_code=fred_code,
                fred_frequency="Unknown", fred_frequency_short="?",
                harmonization="FRED_NATIVE",
                status="FRED_EMPTY", n_obs=0, start=None, end=None, notes=None
            ))
            continue

        s = df_m.set_index("date")["value"]
        s.name = name
        series_monthly[name] = s

        nonmiss = s.dropna()
        audit_rows.append(AuditRow(
            name=name, fred_code=fred_code,
            fred_frequency="Monthly (derived)", fred_frequency_short="m",
                harmonization="FRED_NATIVE",
            status="OK", n_obs=int(nonmiss.shape[0]),
            start=str(nonmiss.index.min().date()) if not nonmiss.empty else None,
            end=str(nonmiss.index.max().date()) if not nonmiss.empty else None,
            notes=None
        ))

    # --- Retry pass for 429 failures (fast first pass, then a small second try only for rate-limited series) ---
    retry_specs = []
    for a in audit_rows:
        if getattr(a, "status", "") == "FRED_ERROR" and isinstance(getattr(a, "notes", None), str) and "HTTP 429" in a.notes:
            retry_specs.append((a.name, a.fred_code))

    if retry_specs:
        # one small pause to give the API a breath without slowing the whole run
        time.sleep(1.5)
        for name, fred_code in retry_specs:
            df_obs, err = fetch_observations(session, fred_code, api_key, timeout=20.0)
            if err is not None:
                continue  # keep the original audit error row
            df_m = to_monthly_last_obs(df_obs)
            if df_m.empty:
                continue
            s = df_m.set_index("date")["value"]
            s.name = name
            series_monthly[name] = s

            # update the existing audit row for this series to OK
            for a in audit_rows:
                if a.name == name and a.fred_code == fred_code:
                    nonmiss = s.dropna()
                    a.status = "OK"
                    a.fred_frequency = "Monthly (derived)"
                    a.fred_frequency_short = "m"
                    a.n_obs = int(nonmiss.shape[0])
                    a.start = str(nonmiss.index.min().date()) if not nonmiss.empty else None
                    a.end = str(nonmiss.index.max().date()) if not nonmiss.empty else None
                    a.notes = None
                    break

    # Monthly index union
    all_dates = set()
    for s in series_monthly.values():
        all_dates.update(list(s.index))
    monthly_index = pd.DatetimeIndex(sorted(all_dates))

    # Wide FRED (fast)
    fred_wide = pd.concat([s.reindex(monthly_index) for s in series_monthly.values()], axis=1)
    fred_wide.insert(0, "date", monthly_index)

    # --- Refinitiv (existing logic) ---
    ref_m = load_refinitiv_monthly(ref_path, monthly_index)
    append_refinitiv_to_audit(ref_m, audit_rows)

    # Combine
    raw_complete = pd.concat([fred_wide.set_index("date"), ref_m.set_index("date")], axis=1).reset_index()

    # Write
    audit_df = pd.DataFrame([a.__dict__ for a in audit_rows])
    audit_df.to_csv(outdir / "fred_fetch_audit.csv", index=False)
    fred_wide.to_csv(outdir / "fred_monthly_wide.csv", index=False)
    raw_complete.to_csv(outdir / "raw_complete_wide.csv", index=False)

    print(f"Wrote: {(outdir / 'fred_fetch_audit.csv').resolve()}")
    print(f"Wrote: {(outdir / 'fred_monthly_wide.csv').resolve()}")
    print(f"Wrote: {(outdir / 'raw_complete_wide.csv').resolve()}")


if __name__ == "__main__":
    main()
