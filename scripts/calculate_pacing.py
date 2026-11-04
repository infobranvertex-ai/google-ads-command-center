"""
Budget pacing, period-over-period summaries and campaign health scoring.

Design decisions (deviations from the original build plan, deliberately):
  * "As of" date = latest date in the data, not today's date. Exports lag; using today()
    makes every campaign look under-paced on the morning the export is a day old.
  * Pacing uses MONTH-TO-DATE spend only, independent of the sidebar date filter.
  * Projected month-end spend uses the last-7-day run rate, so a budget change on the
    20th shows up in the projection instead of being averaged away.
  * Impression-share metrics are impression-weighted, never averaged across days.
"""
from __future__ import annotations

import calendar

import numpy as np
import pandas as pd

GROUP_COLS = ["campaign_id", "campaign_name", "campaign_type", "campaign_status",
              "owner_specialist", "primary_geo", "currency"]

PACE_HIGH, PACE_LOW = 110.0, 90.0


def _safe_div(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(b > 0, a / b, np.nan)


def latest_attributes(df: pd.DataFrame) -> pd.DataFrame:
    """One row per campaign with its most recent name/type/status/owner/budget."""
    cols = [c for c in GROUP_COLS + ["daily_budget", "planned_monthly_spend", "landing_page_url"]
            if c in df.columns]
    return (df.sort_values("date").groupby("campaign_id", as_index=False)[cols[1:]].last()
            if len(df) else pd.DataFrame(columns=cols))


def summarize(df: pd.DataFrame, by: str | list[str] = "campaign_id") -> pd.DataFrame:
    """Aggregate base metrics and recompute ratios from sums."""
    by = [by] if isinstance(by, str) else by
    d = df.copy()
    for m in ["search_impression_share", "search_lost_is_budget", "search_lost_is_rank"]:
        if m in d.columns:
            # weight by impressions only where the metric exists (Search rows)
            w = d["impressions"].where(d[m].notna(), 0)
            d[f"_{m}_w"] = d[m].fillna(0) * w
            d[f"_{m}_n"] = w
    agg = {"cost": "sum", "impressions": "sum", "clicks": "sum",
           "conversions": "sum", "conversion_value": "sum"}
    agg.update({c: "sum" for c in d.columns if c.startswith("_") and c.endswith(("_w", "_n"))})
    out = d.groupby(by, as_index=False).agg(agg)
    for m in ["search_impression_share", "search_lost_is_budget", "search_lost_is_rank"]:
        if f"_{m}_w" in out.columns:
            out[m] = _safe_div(out[f"_{m}_w"], out[f"_{m}_n"])
            out = out.drop(columns=[f"_{m}_w", f"_{m}_n"])
    out = out.rename(columns={"cost": "spend"})
    out["ctr"] = _safe_div(out["clicks"], out["impressions"])
    out["avg_cpc"] = _safe_div(out["spend"], out["clicks"])
    out["conv_rate"] = _safe_div(out["conversions"], out["clicks"])
    out["cpa"] = _safe_div(out["spend"], out["conversions"])
    out["roas"] = _safe_div(out["conversion_value"], out["spend"])
    return out


def merge_targets(campaigns: pd.DataFrame, targets: pd.DataFrame | None) -> pd.DataFrame:
    """Targets win over values embedded in the export. Avoids _x/_y column collisions."""
    c = campaigns.copy()
    for col in ["planned_monthly_spend", "target_cpa", "target_roas"]:
        if col not in c.columns:
            c[col] = np.nan
    if targets is None or targets.empty:
        return c
    t = targets.copy()
    t["campaign_id"] = t["campaign_id"].astype(str)
    c = c.merge(t, on="campaign_id", how="left", suffixes=("", "_tgt"))
    for col in ["planned_monthly_spend", "target_cpa", "target_roas", "owner_specialist"]:
        if f"{col}_tgt" in c.columns:
            override = c[f"{col}_tgt"].notna()
            if col == "owner_specialist":
                override &= c[f"{col}_tgt"].astype(str).str.strip().ne("")
            c.loc[override, col] = c.loc[override, f"{col}_tgt"]
            c = c.drop(columns=f"{col}_tgt")
    return c


def pacing(df: pd.DataFrame, targets: pd.DataFrame | None = None,
           as_of: pd.Timestamp | None = None) -> tuple[pd.DataFrame, dict]:
    """Month-to-date pacing per campaign. Returns (table, context)."""
    as_of = pd.Timestamp(as_of) if as_of is not None else df["date"].max()
    month_start = as_of.replace(day=1)
    days_in_month = calendar.monthrange(as_of.year, as_of.month)[1]
    days_elapsed = as_of.day
    remaining = days_in_month - days_elapsed

    mtd = df[(df["date"] >= month_start) & (df["date"] <= as_of)]
    rr_start = max(month_start, as_of - pd.Timedelta(days=6))
    rr_days = (as_of - rr_start).days + 1
    recent = df[(df["date"] >= rr_start) & (df["date"] <= as_of)]

    attrs = latest_attributes(df)
    m = summarize(mtd)[["campaign_id", "spend", "conversions", "conversion_value", "clicks"]]
    rr = recent.groupby("campaign_id", as_index=False)["cost"].sum().rename(columns={"cost": "_rr"})

    t = attrs.merge(m, on="campaign_id", how="left").merge(rr, on="campaign_id", how="left")
    t[["spend", "conversions", "conversion_value", "clicks", "_rr"]] = \
        t[["spend", "conversions", "conversion_value", "clicks", "_rr"]].fillna(0.0)
    if "planned_monthly_spend" in t.columns:  # embedded value from export; targets override below
        t["planned_monthly_spend"] = t["planned_monthly_spend"].where(t["planned_monthly_spend"] > 0)
    t = merge_targets(t, targets)

    t["expected_spend_to_date"] = t["planned_monthly_spend"] * days_elapsed / days_in_month
    t["pacing_pct"] = _safe_div(t["spend"], t["expected_spend_to_date"]) * 100
    t["run_rate_daily"] = t["_rr"] / rr_days
    t["projected_month_end"] = t["spend"] + t["run_rate_daily"] * remaining
    t["projected_vs_plan_pct"] = _safe_div(t["projected_month_end"], t["planned_monthly_spend"]) * 100
    t["budget_utilization_pct"] = _safe_div(t["spend"], t["planned_monthly_spend"]) * 100
    t["mtd_cpa"] = _safe_div(t["spend"], t["conversions"])
    t["mtd_roas"] = _safe_div(t["conversion_value"], t["spend"])
    # Pro-rated conversion target only exists when both budget and target CPA exist
    t["expected_conversions_to_date"] = _safe_div(t["expected_spend_to_date"], t["target_cpa"])
    t = t.drop(columns="_rr")

    active = t["campaign_status"].str.lower().eq("enabled")
    status = np.select(
        [active & t["spend"].eq(0),
         t["planned_monthly_spend"].isna(),
         t["pacing_pct"] > PACE_HIGH,
         t["pacing_pct"] < PACE_LOW],
        ["Zero spend", "No target", "Overspending", "Underspending"],
        default="On pace",
    )
    t["pacing_status"] = status
    t.loc[~active & t["spend"].eq(0), "pacing_status"] = "Inactive"

    ctx = {"as_of": as_of, "month_start": month_start, "days_elapsed": days_elapsed,
           "days_in_month": days_in_month, "remaining_days": remaining, "run_rate_days": rr_days,
           "early_month": days_elapsed < 7}
    return t.sort_values("spend", ascending=False).reset_index(drop=True), ctx


def period_compare(df: pd.DataFrame, end: pd.Timestamp, days: int) -> pd.DataFrame:
    """Current N days vs previous N days, per campaign, with % deltas."""
    end = pd.Timestamp(end)
    cur_start = end - pd.Timedelta(days=days - 1)
    prev_end = cur_start - pd.Timedelta(days=1)
    prev_start = prev_end - pd.Timedelta(days=days - 1)
    cur = summarize(df[(df["date"] >= cur_start) & (df["date"] <= end)])
    prev = summarize(df[(df["date"] >= prev_start) & (df["date"] <= prev_end)])
    keep = ["campaign_id", "spend", "conversions", "cpa", "roas", "ctr", "conv_rate"]
    out = cur.merge(prev[keep], on="campaign_id", how="left", suffixes=("", "_prev"))
    for k in ["spend", "conversions", "cpa", "roas", "ctr", "conv_rate"]:
        out[f"{k}_chg_pct"] = (_safe_div(out[k], out[f"{k}_prev"]) - 1) * 100
    return out


def health_scores(window: pd.DataFrame, pace: pd.DataFrame) -> pd.DataFrame:
    """
    Transparent 0-100 health score. Starts at 100, deducts:
      Efficiency (max -40): CPA over target, or ROAS under target (pro-rata, capped)
      Zero conversions on material spend: -30
      Pacing outside 90-110% (max -15): 1 point per 2pp outside band
      Budget-lost impression share (max -15): 0.5 point per pp lost to budget
    Campaigns without targets are scored on the parts that can be measured.
    """
    h = window.merge(
        pace[["campaign_id", "pacing_pct", "pacing_status", "target_cpa", "target_roas",
              "planned_monthly_spend"]],
        on="campaign_id", how="left")
    acct_cpa = _safe_div(h["spend"].sum(), h["conversions"].sum()).item()
    material = h["spend"] >= (2 * acct_cpa if np.isfinite(acct_cpa) else h["spend"].median())

    eff = np.zeros(len(h))
    over_cpa = _safe_div(h["cpa"], h["target_cpa"]) - 1
    eff = np.where(np.isfinite(over_cpa) & (over_cpa > 0), np.minimum(over_cpa * 100, 40), eff)
    under_roas = 1 - _safe_div(h["roas"], h["target_roas"])
    eff = np.where(np.isfinite(under_roas) & (under_roas > 0),
                   np.maximum(eff, np.minimum(under_roas * 100, 40)), eff)

    zero = np.where(material & h["conversions"].eq(0), 30, 0)
    p = h["pacing_pct"].to_numpy(dtype=float)
    pace_pen = np.where(np.isfinite(p),
                        np.minimum(np.maximum(np.maximum(p - PACE_HIGH, PACE_LOW - p), 0) / 2, 15), 0)
    lis = h.get("search_lost_is_budget", pd.Series(np.nan, index=h.index)).to_numpy(dtype=float)
    lis_pen = np.where(np.isfinite(lis), np.minimum(lis * 100 * 0.5, 15), 0)

    h["pen_efficiency"] = eff.round(1)
    h["pen_zero_conv"] = zero
    h["pen_pacing"] = pace_pen.round(1)
    h["pen_budget_lost_is"] = lis_pen.round(1)
    h["health_score"] = (100 - eff - zero - pace_pen - lis_pen).clip(0, 100).round(0)
    h["health_band"] = pd.cut(h["health_score"], [-1, 49, 74, 100], labels=["Poor", "Watch", "Healthy"])
    return h.sort_values(["health_score", "spend"], ascending=[True, False]).reset_index(drop=True)
