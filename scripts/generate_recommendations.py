"""
Rule-based action queue, ranked by priority then money at stake.

Every rule has a materiality guard so 300 campaigns don't produce 300 alerts.
The thresholds live in RULES so they can be tuned in one place (and are shown in the app).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

RULES = {
    "zero_conv_min_spend_x_cpa": 2.0,   # zero-conv flag needs spend >= 2x target CPA (or account CPA)
    "pace_critical": 130.0,
    "pace_low_high": 70.0,
    "cpa_over_target": 0.20,            # CPA >20% above target
    "cpa_spike": 0.20,                  # CPA up >20% vs previous period...
    "cpa_spike_z": 3.0,                 # ...AND conversion drop >= 3 std devs (2 sigma x 300 campaigns = ~7 false alarms)
    "min_conv_for_trend": 20,           # need >=20 conv in BOTH periods; below that a 20% CPA swing is noise
    "lost_is_budget_high": 0.20,
    "lost_is_budget_strong_roas": 0.10,
    "strong_roas_x_target": 1.20,
    "min_impr_ctr": 1000,
    "ctr_below_median": 0.50,           # CTR < 50% of campaign-type median
    "min_clicks_cvr": 100,
    "cvr_below_median": 0.50,
}

PRIORITY_ORDER = {"Critical": 0, "High": 1, "Medium": 2, "Low": 3}


def _fmt(x, kind="num"):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "n/a"
    return {"pct": f"{x:.0%}", "pp": f"{x:.0f}%", "x": f"{x:.2f}x"}.get(kind, f"{x:,.0f}")


def campaign_actions(health: pd.DataFrame, pace: pd.DataFrame, trend: pd.DataFrame | None = None,
                     rules: dict | None = None) -> pd.DataFrame:
    """
    health: output of health_scores() for the selected window (30 days recommended)
    pace:   output of pacing() (month-to-date)
    trend:  output of period_compare() for CPA-spike detection (optional)
    """
    r = {**RULES, **(rules or {})}
    h = health.copy()
    p = pace.set_index("campaign_id")
    acct_cpa = h["spend"].sum() / h["conversions"].sum() if h["conversions"].sum() > 0 else np.nan
    type_ctr = h[h["impressions"] >= r["min_impr_ctr"]].groupby("campaign_type")["ctr"].median()
    type_cvr = h[h["clicks"] >= r["min_clicks_cvr"]].groupby("campaign_type")["conv_rate"].median()
    tr = trend.set_index("campaign_id") if trend is not None and len(trend) else None

    rows = []

    def add(row, priority, issue, evidence, action, at_stake):
        rows.append({"Priority": priority, "Campaign": row["campaign_name"],
                     "Type": row.get("campaign_type"), "Owner": row.get("owner_specialist"),
                     "Issue": issue, "Evidence": evidence, "Recommended action": action,
                     "Money at stake": float(at_stake) if np.isfinite(at_stake) else 0.0,
                     "campaign_id": row["campaign_id"]})

    for _, row in h.iterrows():
        cid = row["campaign_id"]
        pr = p.loc[cid] if cid in p.index else None
        tcpa, troas = row.get("target_cpa"), row.get("target_roas")
        ref_cpa = tcpa if pd.notna(tcpa) else acct_cpa
        spend, conv = row["spend"], row["conversions"]
        cpa, roas = row["cpa"], row["roas"]
        pace_pct = pr["pacing_pct"] if pr is not None else np.nan
        lis = row.get("search_lost_is_budget", np.nan)

        # Critical: material spend, zero conversions
        if conv == 0 and pd.notna(ref_cpa) and spend >= r["zero_conv_min_spend_x_cpa"] * ref_cpa:
            add(row, "Critical", "Material spend, zero conversions",
                f"Spend {_fmt(spend)} = {spend / ref_cpa:.1f}x {'target' if pd.notna(tcpa) else 'account'} CPA, 0 conv.",
                "Check conversion tracking first; then pause or restrict the worst keywords/ad groups.",
                spend)

        # Critical: heavy overpacing AND inefficient
        if pd.notna(pace_pct) and pace_pct > r["pace_critical"] and pd.notna(tcpa) and pd.notna(cpa) and cpa > tcpa:
            overspend = pr["projected_month_end"] - pr["planned_monthly_spend"]
            add(row, "Critical", "Overpacing with CPA above target",
                f"Pacing {_fmt(pace_pct, 'pp')}, CPA {_fmt(cpa)} vs target {_fmt(tcpa)}.",
                "Cut daily budget toward plan or tighten tCPA; do not let inefficient spend run to month-end.",
                max(overspend, 0))

        # High: underpacing AND behind conversion goal
        if pr is not None and pd.notna(pace_pct) and pace_pct < r["pace_low_high"]:
            exp_conv = pr.get("expected_conversions_to_date", np.nan)
            if pd.notna(exp_conv) and pr["conversions"] < exp_conv:
                shortfall = pr["expected_spend_to_date"] - pr["spend"]
                add(row, "High", "Underpacing and behind conversion goal",
                    f"Pacing {_fmt(pace_pct, 'pp')}; {pr['conversions']:.0f} conv vs {exp_conv:.0f} expected MTD.",
                    "Diagnose delivery: approvals, bid caps, audience/geo limits, low search volume.",
                    shortfall)

        # High: CPA over target
        if pd.notna(tcpa) and pd.notna(cpa) and cpa > tcpa * (1 + r["cpa_over_target"]):
            excess = spend - conv * tcpa
            add(row, "High", "CPA above target",
                f"CPA {_fmt(cpa)} vs target {_fmt(tcpa)} (+{cpa / tcpa - 1:.0%}).",
                "Review search terms, pause bleeding keywords, check landing page and bid strategy.",
                excess)

        # High: ROAS under target (ROAS-managed campaigns)
        if pd.notna(troas) and pd.notna(roas) and spend > 0 and roas < troas / (1 + r["cpa_over_target"]):
            lost_value = spend * troas - row["conversion_value"]
            add(row, "High", "ROAS below target",
                f"ROAS {_fmt(roas, 'x')} vs target {_fmt(troas, 'x')}.",
                "Check product/asset-group mix, feed quality, and tROAS setting.",
                lost_value)

        # High: CPA spike vs previous period (noise-guarded)
        if tr is not None and cid in tr.index:
            t = tr.loc[cid]
            # Poisson guard: conversions this period vs what last period's CPA would have bought.
            expected = t["spend"] / t["cpa_prev"] if pd.notna(t.get("cpa_prev")) and t["cpa_prev"] > 0 else np.nan
            z = (expected - t["conversions"]) / np.sqrt(expected) if pd.notna(expected) and expected > 0 else 0
            if (t["conversions"] >= r["min_conv_for_trend"] and t.get("conversions_prev", 0) >= r["min_conv_for_trend"]
                    and pd.notna(t["cpa_chg_pct"]) and t["cpa_chg_pct"] > r["cpa_spike"] * 100
                    and z >= r["cpa_spike_z"]):
                add(row, "High", "CPA spike vs previous period",
                    f"CPA {_fmt(t['cpa'])} vs {_fmt(t['cpa_prev'])} (+{t['cpa_chg_pct']:.0f}%).",
                    "Check change history, auction insights and search-term drift for the period.",
                    t["spend"] - t["conversions"] * t["cpa_prev"])

        # Budget-limited: split by efficiency, so we never recommend scaling a bad campaign
        if pd.notna(lis):
            efficient = ((pd.notna(tcpa) and pd.notna(cpa) and cpa <= tcpa) or
                         (pd.notna(troas) and pd.notna(roas) and roas >= troas))
            strong = pd.notna(troas) and pd.notna(roas) and roas >= troas * r["strong_roas_x_target"]
            if lis > r["lost_is_budget_high"]:
                if efficient:
                    add(row, "High", "Efficient but budget-limited",
                        f"{_fmt(lis, 'pct')} impression share lost to budget at target efficiency.",
                        "Raise budget in 15-20% steps; recheck CPA/ROAS after 7 days.",
                        spend * lis / max(1 - lis, 0.01))
                else:
                    add(row, "Medium", "Budget-limited but not efficient",
                        f"{_fmt(lis, 'pct')} IS lost to budget; efficiency not at target (or no target set).",
                        "Fix efficiency or set a target before adding budget. More budget here buys more of the same.",
                        0)
            elif strong and lis > r["lost_is_budget_strong_roas"]:
                add(row, "Low", "Strong ROAS with budget headroom",
                    f"ROAS {_fmt(roas, 'x')} vs target {_fmt(troas, 'x')}; {_fmt(lis, 'pct')} IS lost to budget.",
                    "Test a gradual budget increase.", spend * lis)

        # Medium: CTR well below campaign-type median
        ct = row.get("campaign_type")
        if row["impressions"] >= r["min_impr_ctr"] and ct in type_ctr.index and pd.notna(row["ctr"]):
            if row["ctr"] < type_ctr[ct] * r["ctr_below_median"]:
                add(row, "Medium", "CTR far below peer campaigns",
                    f"CTR {row['ctr']:.2%} vs {ct} median {type_ctr[ct]:.2%}.",
                    "Test new RSA headlines/assets; check keyword-to-ad relevance.", 0)

        # Medium: healthy CTR, weak conversion rate -> landing page problem
        if (row["clicks"] >= r["min_clicks_cvr"] and ct in type_cvr.index and ct in type_ctr.index
                and pd.notna(row["conv_rate"]) and row["conv_rate"] < type_cvr[ct] * r["cvr_below_median"]
                and pd.notna(row["ctr"]) and row["ctr"] >= type_ctr[ct]):
            add(row, "Medium", "Good CTR, weak conversion rate",
                f"CVR {row['conv_rate']:.2%} vs {ct} median {type_cvr[ct]:.2%}; CTR at/above median.",
                "Landing page or offer mismatch: check page speed, form, price/stock, message match.",
                spend * (1 - row["conv_rate"] / type_cvr[ct]) if type_cvr[ct] > 0 else 0)

    # Pacing statuses that need a human regardless of efficiency.
    # If most campaigns have no plan, that's a setup gap (shown as a banner), not 300 tasks.
    no_plan_share = (pace["pacing_status"] == "No target").mean() if len(pace) else 0
    for _, pr in pace.iterrows():
        if pr["pacing_status"] == "Zero spend":
            add(pr, "High", "Enabled campaign with zero spend MTD", "No spend this month despite Enabled status.",
                "Check billing, approvals, budget, targeting and bid floors.", pr.get("expected_spend_to_date", 0) or 0)
        elif pr["pacing_status"] == "No target" and pr["spend"] > 0 and no_plan_share <= 0.5:
            add(pr, "Low", "No planned budget", f"Spent {_fmt(pr['spend'])} MTD with no plan on file.",
                "Add planned_monthly_spend (and target CPA/ROAS) to campaign_targets.csv.", 0)

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["_p"] = out["Priority"].map(PRIORITY_ORDER)
    out = out.sort_values(["_p", "Money at stake"], ascending=[True, False]).drop(columns="_p")
    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Keyword / search term / ad level
# ---------------------------------------------------------------------------

def _ratios(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    with np.errstate(divide="ignore", invalid="ignore"):
        d["ctr"] = np.where(d["impressions"] > 0, d["clicks"] / d["impressions"], np.nan) if "impressions" in d else np.nan
        d["conv_rate"] = np.where(d["clicks"] > 0, d["conversions"] / d["clicks"], np.nan)
        d["cpa"] = np.where(d["conversions"] > 0, d["cost"] / d["conversions"], np.nan)
        if "conversion_value" in d:
            d["roas"] = np.where(d["cost"] > 0, d["conversion_value"] / d["cost"], np.nan)
    return d


def keyword_views(kw: pd.DataFrame, ref_cpa: float) -> dict[str, pd.DataFrame]:
    keys = [c for c in ["campaign_name", "ad_group", "keyword", "match_type"] if c in kw.columns]
    agg = {c: "sum" for c in ["cost", "impressions", "clicks", "conversions", "conversion_value"] if c in kw.columns}
    if "quality_score" in kw.columns:
        agg["quality_score"] = "last"
    k = _ratios(kw.groupby(keys, as_index=False, dropna=False).agg(agg))
    thr = 2 * ref_cpa if np.isfinite(ref_cpa) else k["cost"].quantile(0.9)
    views = {
        "Top converters": k[k["conversions"] > 0].sort_values(["conversions", "cpa"], ascending=[False, True]).head(50),
        "High spend, zero conversions": k[(k["conversions"] == 0) & (k["cost"] >= thr)].sort_values("cost", ascending=False),
        "Low CTR (>=1,000 impr.)": k[(k["impressions"] >= 1000) & (k["ctr"] < k.loc[k["impressions"] >= 1000, "ctr"].median() * 0.5)]
            .sort_values("impressions", ascending=False),
    }
    if "quality_score" in k.columns:
        views["Quality Score < 5 (>100 impr.)"] = k[(k["quality_score"] < 5) & (k["impressions"] > 100)].sort_values("cost", ascending=False)
    return views


def search_term_views(st_df: pd.DataFrame, ref_cpa: float) -> dict[str, pd.DataFrame]:
    keys = [c for c in ["search_term", "campaign_name", "ad_group", "term_status"] if c in st_df.columns]
    agg = {c: "sum" for c in ["cost", "impressions", "clicks", "conversions", "conversion_value"] if c in st_df.columns}
    s = _ratios(st_df.groupby(keys, as_index=False, dropna=False).agg(agg))
    thr = ref_cpa if np.isfinite(ref_cpa) else s["cost"].quantile(0.9)
    not_added = ~s.get("term_status", pd.Series("None", index=s.index)).astype(str).str.lower().isin(["added", "excluded"])
    return {
        "Negative keyword candidates": s[(s["conversions"] == 0) & (s["cost"] >= thr) & not_added].sort_values("cost", ascending=False),
        "Promote to exact match": s[(s["conversions"] >= 2) & not_added].sort_values("conversions", ascending=False),
    }


def ad_views(ads: pd.DataFrame, ref_cpa: float, min_clicks: int = 100) -> dict[str, pd.DataFrame]:
    keys = [c for c in ["campaign_name", "ad_group", "ad_id", "ad_type", "headline"] if c in ads.columns]
    agg = {c: "sum" for c in ["cost", "impressions", "clicks", "conversions", "conversion_value"] if c in ads.columns}
    a = _ratios(ads.groupby(keys, as_index=False, dropna=False).agg(agg))
    thr = 2 * ref_cpa if np.isfinite(ref_cpa) else a["cost"].quantile(0.9)
    # Compare ads only within their own ad group, and only with enough clicks
    eligible = a[a["clicks"] >= min_clicks].copy()
    eligible["ag_cvr"] = eligible.groupby(["campaign_name", "ad_group"])["conv_rate"].transform("median")
    eligible["ads_in_test"] = eligible.groupby(["campaign_name", "ad_group"])["clicks"].transform("size")
    eligible = eligible[eligible["ads_in_test"] >= 2]
    eligible["cvr_vs_ad_group"] = eligible["conv_rate"] / eligible["ag_cvr"].replace(0, np.nan) - 1
    return {
        "Winning variations (vs own ad group)": eligible.sort_values("cvr_vs_ad_group", ascending=False).head(30),
        "Losing variations (vs own ad group)": eligible.sort_values("cvr_vs_ad_group").head(30),
        "High spend, zero conversions": a[(a["conversions"] == 0) & (a["cost"] >= thr)].sort_values("cost", ascending=False),
    }
