"""
Load and normalise Google Ads exports into the master-sheet schema.

Accepts either:
  * the master-sheet column names from the build plan (date, campaign_id, cost ...), or
  * raw Google Ads UI export headers (Day, Campaign ID, Cost, Conv. value, Impr. ...).

Percentages are stored as fractions (0.25 = 25%). Google's "< 10%" / "> 90%"
impression-share buckets are converted to 0.05 / 0.95 and flagged.

Run as a script to write data/processed/campaign_performance.csv:
    python scripts/process_data.py data/raw/google_ads_campaign_daily.csv
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Column aliases (lower-cased, whitespace-collapsed header -> canonical name)
# ---------------------------------------------------------------------------
ALIASES: dict[str, str] = {
    # dates / ids
    "day": "date", "date": "date", "segments.date": "date",
    "customer id": "account_id", "account id": "account_id", "account_id": "account_id",
    "account": "account_name", "account name": "account_name", "account_name": "account_name",
    "campaign id": "campaign_id", "campaign_id": "campaign_id", "campaign.id": "campaign_id",
    "campaign": "campaign_name", "campaign name": "campaign_name", "campaign_name": "campaign_name",
    "campaign type": "campaign_type", "campaign_type": "campaign_type",
    "advertising channel type": "campaign_type",
    "campaign status": "campaign_status", "campaign state": "campaign_status",
    "campaign_status": "campaign_status", "status": "campaign_status",
    "budget": "daily_budget", "daily budget": "daily_budget", "daily_budget": "daily_budget",
    "planned_monthly_spend": "planned_monthly_spend", "planned monthly spend": "planned_monthly_spend",
    "currency code": "currency", "currency": "currency",
    # metrics
    "impr.": "impressions", "impressions": "impressions",
    "clicks": "clicks", "interactions": "clicks",
    "cost": "cost", "cost_micros": "cost_micros",
    "conversions": "conversions", "conv.": "conversions",
    "conv. value": "conversion_value", "conversion value": "conversion_value",
    "conversion_value": "conversion_value", "all conv. value": "conversion_value",
    "search impr. share": "search_impression_share",
    "search impression share": "search_impression_share",
    "search_impression_share": "search_impression_share",
    "search lost is (budget)": "search_lost_is_budget",
    "search lost impression share (budget)": "search_lost_is_budget",
    "search_lost_is_budget": "search_lost_is_budget",
    "search lost is (rank)": "search_lost_is_rank",
    "search lost impression share (rank)": "search_lost_is_rank",
    "search_lost_is_rank": "search_lost_is_rank",
    "final url": "landing_page_url", "landing page": "landing_page_url",
    "landing_page_url": "landing_page_url", "expanded landing page": "landing_page_url",
    "primary_geo": "primary_geo", "geo": "primary_geo",
    "owner_specialist": "owner_specialist", "owner": "owner_specialist", "specialist": "owner_specialist",
    "target_cpa": "target_cpa", "target cpa": "target_cpa",
    "target_roas": "target_roas", "target roas": "target_roas",
    # keyword / search-term / ad level
    "keyword": "keyword", "search keyword": "keyword",
    "match type": "match_type", "search keyword match type": "match_type", "match_type": "match_type",
    "ad group": "ad_group", "ad_group": "ad_group",
    "quality score": "quality_score", "quality_score": "quality_score",
    "search term": "search_term", "search_term": "search_term",
    "added/excluded": "term_status", "term_status": "term_status",
    "ad id": "ad_id", "ad_id": "ad_id",
    "ad type": "ad_type", "ad_type": "ad_type",
    "headline 1": "headline", "headline": "headline", "ad": "headline",
}

# Derived ratio columns are always recomputed from base metrics, never trusted from the export.
DROP_IF_PRESENT = ["ctr", "average_cpc", "avg. cpc", "cost_per_conversion", "cost / conv.",
                   "conversion_rate", "conv. rate", "roas", "conv. value / cost"]

CAMPAIGN_REQUIRED = ["date", "campaign_id", "campaign_name", "impressions", "clicks", "cost", "conversions"]
KEYWORD_REQUIRED = ["keyword", "campaign_name", "impressions", "clicks", "cost", "conversions"]
SEARCH_TERM_REQUIRED = ["search_term", "campaign_name", "clicks", "cost", "conversions"]
AD_REQUIRED = ["campaign_name", "ad_group", "impressions", "clicks", "cost", "conversions"]

NUMERIC = ["daily_budget", "planned_monthly_spend", "impressions", "clicks", "cost", "cost_micros",
           "conversions", "conversion_value", "target_cpa", "target_roas", "quality_score"]
PERCENT = ["search_impression_share", "search_lost_is_budget", "search_lost_is_rank"]
TEXT_DEFAULTS = {
    "account_id": "unknown", "account_name": "unknown", "campaign_type": "Unknown",
    "campaign_status": "Enabled", "currency": "N/A", "landing_page_url": "(not set)",
    "primary_geo": "(not set)", "owner_specialist": "Unassigned",
}


def _canon(col: str) -> str:
    return re.sub(r"\s+", " ", str(col).strip().lower())


def _to_number(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(float)
    cleaned = (s.astype(str).str.strip()
               .replace({"--": np.nan, "": np.nan, " --": np.nan, "nan": np.nan, "None": np.nan})
               .str.replace(r"[,\s]", "", regex=True)
               .str.replace(r"^[^\d\-.]+", "", regex=True))  # strip leading currency symbols
    return pd.to_numeric(cleaned, errors="coerce")


def _to_fraction(s: pd.Series) -> tuple[pd.Series, int]:
    """Parse '45.3%', '< 10%', '> 90%', 0.453 or 45.3 into a 0-1 fraction."""
    if pd.api.types.is_numeric_dtype(s):
        v = s.astype(float)
        return (v / 100 if v.max(skipna=True) > 1.0 else v), 0
    raw = s.astype(str).str.strip()
    bucketed = raw.str.startswith("<") | raw.str.startswith(">")
    out = pd.Series(np.nan, index=s.index, dtype=float)
    out[raw.str.startswith("<")] = 0.05
    out[raw.str.startswith(">")] = 0.95
    rest = ~bucketed
    nums = pd.to_numeric(raw[rest].str.replace("%", "", regex=False).str.replace(",", "", regex=False),
                         errors="coerce")
    has_pct = raw[rest].str.contains("%", regex=False)
    nums = np.where(has_pct | (nums > 1.0), nums / 100, nums)
    out[rest] = nums
    return out, int(bucketed.sum())


def normalize(df: pd.DataFrame, required: list[str] | None = None) -> tuple[pd.DataFrame, list[str]]:
    """Return (clean_df, warnings). Raises ValueError if required columns are missing."""
    warnings: list[str] = []
    df = df.copy()

    # Google Ads UI exports often have 2 title rows above the header and a "Total:" footer.
    df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed")]
    drop = [c for c in df.columns if _canon(c) in DROP_IF_PRESENT]
    df = df.drop(columns=drop)

    rename, seen = {}, set()
    for c in df.columns:
        target = ALIASES.get(_canon(c), _canon(c).replace(" ", "_"))
        if target in seen:  # keep first occurrence of duplicated meanings
            continue
        seen.add(target)
        rename[c] = target
    df = df[list(rename)].rename(columns=rename)

    if required:
        missing = [c for c in required if c not in df.columns]
        if "cost" in missing and "cost_micros" in df.columns:
            missing.remove("cost")
        if missing:
            raise ValueError(f"Missing required columns: {', '.join(missing)}")

    # Drop footer/total rows
    first_text = df.columns[0]
    df = df[~df[first_text].astype(str).str.lower().str.startswith("total")]

    for col in NUMERIC:
        if col in df.columns:
            df[col] = _to_number(df[col])
    if "cost" not in df.columns and "cost_micros" in df.columns:
        df["cost"] = df["cost_micros"] / 1_000_000
    for col in ["impressions", "clicks", "cost", "conversions", "conversion_value"]:
        if col in df.columns:
            df[col] = df[col].fillna(0.0)
        elif col == "conversion_value":
            df[col] = 0.0

    bucket_total = 0
    for col in PERCENT:
        if col in df.columns:
            df[col], n = _to_fraction(df[col])
            bucket_total += n
    if bucket_total:
        warnings.append(f"{bucket_total} impression-share cells were bucketed ('< 10%' / '> 90%') "
                        "and set to 5% / 95%. Treat those values as approximate.")

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        bad = df["date"].isna().sum()
        if bad:
            warnings.append(f"Dropped {bad} rows with unreadable dates.")
            df = df[df["date"].notna()]

    for col in ["campaign_id", "account_id", "ad_id"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(r"\.0$", "", regex=True).str.strip()
    if "campaign_id" not in df.columns and "campaign_name" in df.columns:
        df["campaign_id"] = df["campaign_name"]
        warnings.append("No campaign ID column: using campaign name as ID. Renamed campaigns will split.")

    for col, default in TEXT_DEFAULTS.items():
        if col in df.columns:
            df[col] = df[col].fillna(default).replace({"": default, "--": default, "nan": default})
        elif "date" in df.columns:  # campaign-daily data: guarantee the column exists
            df[col] = default
    if "campaign_status" in df.columns:
        df["campaign_status"] = df["campaign_status"].astype(str).str.strip().str.capitalize()

    neg = (df[["impressions", "clicks", "cost"]] < 0).any(axis=1).sum() if {"impressions", "clicks", "cost"} <= set(df.columns) else 0
    if neg:
        warnings.append(f"{neg} rows contain negative metrics (usually invalid-click credits).")

    return df.reset_index(drop=True), warnings


def _header_row(lines: list[str], sep: str) -> int:
    known = set(ALIASES)
    for i, line in enumerate(lines[:10]):
        cells = [_canon(c.strip('"')) for c in line.split(sep)]
        if sum(c in known for c in cells) >= 3:
            return i
    return 0


def read_any(path_or_buffer) -> pd.DataFrame:
    """
    Read CSV / TSV / XLSX exports. Handles Google Ads quirks:
      * 1-2 report-title lines above the header
      * "CSV (Excel)" downloads that are UTF-16, tab-separated
      * thousands separators
    """
    import io

    name = getattr(path_or_buffer, "name", str(path_or_buffer)).lower()
    if name.endswith((".xlsx", ".xls")):
        probe = pd.read_excel(path_or_buffer, header=None, nrows=10)
        lines = ["\t".join(map(str, r)) for r in probe.fillna("").values]
        if hasattr(path_or_buffer, "seek"):
            path_or_buffer.seek(0)
        return pd.read_excel(path_or_buffer, skiprows=_header_row(lines, "\t"))

    raw = path_or_buffer.read() if hasattr(path_or_buffer, "read") else Path(path_or_buffer).read_bytes()
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
        text = raw.decode("utf-16")
    else:
        try:
            text = raw.decode("utf-8-sig")
        except UnicodeDecodeError:
            text = raw.decode("latin-1")
    lines = text.splitlines()
    sep = "\t" if sum(l.count("\t") for l in lines[:10]) > sum(l.count(",") for l in lines[:10]) else ","
    skip = _header_row(lines, sep)
    return pd.read_csv(io.StringIO(text), sep=sep, skiprows=skip, thousands=",", low_memory=False)


def apply_target_attributes(df: pd.DataFrame, targets: pd.DataFrame | None) -> pd.DataFrame:
    """Owner and geo from the targets file override the export (the API has neither)."""
    if targets is None or targets.empty:
        return df
    df = df.copy()
    for col in ["owner_specialist", "primary_geo"]:
        if col in targets.columns:
            m = targets.dropna(subset=[col]).set_index("campaign_id")[col]
            mapped = df["campaign_id"].map(m)
            df[col] = mapped.where(mapped.notna(), df.get(col))
    return df


def normalize_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Targets file: campaign_id + planned_monthly_spend / target_cpa / target_roas (+ optional owner)."""
    t, _ = normalize(df, required=["campaign_id"])
    keep = [c for c in ["campaign_id", "planned_monthly_spend", "target_cpa", "target_roas",
                        "owner_specialist", "primary_geo"] if c in t.columns]
    t = t[keep].drop_duplicates("campaign_id", keep="last")
    for c in ["target_cpa", "target_roas", "planned_monthly_spend"]:
        if c in t.columns:
            t.loc[t[c] <= 0, c] = np.nan  # 0 in the template means "no target"
    return t


if __name__ == "__main__":
    src = Path(sys.argv[1] if len(sys.argv) > 1 else "data/raw/google_ads_campaign_daily.csv")
    out = Path("data/processed/campaign_performance.csv")
    if not src.exists():
        sys.exit(f"Input not found: {src}")
    clean, warns = normalize(read_any(src), required=CAMPAIGN_REQUIRED)
    out.parent.mkdir(parents=True, exist_ok=True)
    clean.to_csv(out, index=False)
    for w in warns:
        print("WARNING:", w)
    print(f"Processed {len(clean):,} rows, {clean['campaign_id'].nunique():,} campaigns -> {out}")
