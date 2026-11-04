"""
Google Ads Command Center — Streamlit dashboard.

Run locally:   streamlit run app/streamlit_app.py
Data sources (first found wins, per dataset):
  1. File uploaded in the sidebar (never written to disk or GitHub)
  2. data/raw/<file>      (local / private use; gitignored)
  3. data/sample/<file>   (anonymised demo data shipped with the repo)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.calculate_pacing import (health_scores, latest_attributes, pacing,  # noqa: E402
                                      period_compare, summarize)
from scripts.generate_recommendations import (RULES, ad_views, campaign_actions,  # noqa: E402
                                              keyword_views, search_term_views)
from scripts.process_data import (AD_REQUIRED, CAMPAIGN_REQUIRED, KEYWORD_REQUIRED,  # noqa: E402
                                  SEARCH_TERM_REQUIRED, apply_target_attributes, normalize,
                                  normalize_targets, read_any)

st.set_page_config(page_title="Google Ads Command Center", page_icon="📊", layout="wide")

FILES = {
    "campaign": "google_ads_campaign_daily.csv",
    "targets": "campaign_targets.csv",
    "keywords": "keywords.csv",
    "search_terms": "search_terms.csv",
    "ads": "ads.csv",
}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def _local_path(kind: str) -> tuple[Path | None, str]:
    folders = [(ROOT / "data" / "raw", "local file")]
    if kind == "targets":
        folders.append((ROOT / "data" / "targets", "local file"))
    folders.append((ROOT / "data" / "sample", "SAMPLE data"))
    for folder, label in folders:
        p = folder / FILES[kind]
        if p.exists():
            return p, label
    return None, "none"


@st.cache_data(show_spinner="Reading data…")
def _load(kind: str, content: bytes | None, name: str | None, path: str | None):
    import io
    if content is not None:
        buf = io.BytesIO(content)
        buf.name = name
        raw = read_any(buf)
    elif path:
        raw = read_any(path)
    else:
        return None, []
    if kind == "targets":
        return normalize_targets(raw), []
    required = {"campaign": CAMPAIGN_REQUIRED, "keywords": KEYWORD_REQUIRED,
                "search_terms": SEARCH_TERM_REQUIRED, "ads": AD_REQUIRED}[kind]
    return normalize(raw, required=required)


def load(kind: str, upload, allow_sample: bool = True) -> tuple[pd.DataFrame | None, list[str], str]:
    try:
        if upload is not None:
            df, warns = _load(kind, upload.getvalue(), upload.name, None)
            return df, warns, f"uploaded: {upload.name}"
        path, label = _local_path(kind)
        if label == "SAMPLE data" and not allow_sample:  # never mix demo data with real data
            return None, [], "none"
        df, warns = _load(kind, None, None, str(path) if path else None)
        return df, warns, label
    except ValueError as e:
        st.sidebar.error(f"{kind}: {e}")
        return None, [], "error"


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------
def money(x, cur=""):
    if x is None or not np.isfinite(x):
        return "n/a"
    sym = {"INR": "₹", "USD": "$", "EUR": "€", "GBP": "£", "JPY": "¥", "CNY": "¥"}.get(cur, "")
    if abs(x) >= 1e7 and cur == "INR":
        return f"{sym}{x / 1e7:,.2f} Cr"
    if abs(x) >= 1e5 and cur == "INR":
        return f"{sym}{x / 1e5:,.2f} L"
    return f"{sym}{x:,.0f}"


def delta(cur, prev, invert=False):
    if prev is None or not np.isfinite(prev) or prev == 0 or not np.isfinite(cur):
        return None
    return f"{(cur / prev - 1):+.1%}"


NUM = st.column_config.NumberColumn
PCT = lambda label: st.column_config.NumberColumn(label, format="%.1f%%")  # noqa: E731


def show(df: pd.DataFrame, cols: dict[str, str], pct_cols=(), frac_cols=(), height=None, key=None):
    """Display selected columns with readable labels. frac_cols are 0-1 values shown as %."""
    use = [c for c in cols if c in df.columns]
    d = df[use].copy()
    for c in frac_cols:
        if c in d.columns:
            d[c] = d[c] * 100
    cfg = {}
    for c in use:
        label = cols[c]
        if c in pct_cols or c in frac_cols:
            cfg[c] = PCT(label)
        elif pd.api.types.is_float_dtype(d[c]) or pd.api.types.is_integer_dtype(d[c]):
            cfg[c] = NUM(label, format="%.2f" if c in ("roas", "mtd_roas", "target_roas", "roas_prev") else "%.0f")
        else:
            cfg[c] = st.column_config.TextColumn(label)
    st.dataframe(d, column_config=cfg, use_container_width=True, hide_index=True, height=height, key=key)


# ---------------------------------------------------------------------------
# Sidebar: data
# ---------------------------------------------------------------------------
st.sidebar.title("📊 Command Center")
with st.sidebar.expander("Data sources", expanded=False):
    up_campaign = st.file_uploader("Campaign daily export (required)", type=["csv", "xlsx"], key="u_c")
    up_targets = st.file_uploader("Campaign targets", type=["csv", "xlsx"], key="u_t")
    up_kw = st.file_uploader("Keyword report (optional)", type=["csv", "xlsx"], key="u_k")
    up_st = st.file_uploader("Search terms report (optional)", type=["csv", "xlsx"], key="u_s")
    up_ads = st.file_uploader("Ad report (optional)", type=["csv", "xlsx"], key="u_a")
    st.caption("Uploads stay in this browser session only. Nothing is saved to GitHub.")

df_all, warns, src = load("campaign", up_campaign)
USING_SAMPLE = "SAMPLE" in src
targets, _, src_t = load("targets", up_targets, allow_sample=USING_SAMPLE)
if df_all is None or df_all.empty:
    st.error("No campaign data found. Upload a campaign-level daily export in the sidebar.")
    st.stop()
df_all = apply_target_attributes(df_all, targets)

# ---------------------------------------------------------------------------
# Sidebar: filters
# ---------------------------------------------------------------------------
st.sidebar.header("Filters")

currencies = sorted(df_all["currency"].dropna().unique())
if len(currencies) > 1:
    cur = st.sidebar.selectbox("Currency (one at a time — never summed)", currencies)
    df_all = df_all[df_all["currency"] == cur]
else:
    cur = currencies[0] if currencies else ""

max_date, min_date = df_all["date"].max(), df_all["date"].min()
date_range = st.sidebar.date_input(
    "Performance window", value=(max(min_date, max_date - pd.Timedelta(days=29)), max_date),
    min_value=min_date, max_value=max_date,
    help="Drives Overview, Health, Landing Pages and the efficiency rules. "
         "Budget Pacing always uses month-to-date, regardless of this window.")
if not isinstance(date_range, (list, tuple)) or len(date_range) != 2:
    st.sidebar.info("Pick an end date.")
    st.stop()
w_start, w_end = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])


def multi(label, col, default_all=True):
    opts = sorted(df_all[col].dropna().astype(str).unique())
    return st.sidebar.multiselect(label, opts, default=opts if default_all and len(opts) <= 12 else [])


f_type = multi("Campaign type", "campaign_type")
f_status = multi("Campaign status", "campaign_status")
f_owner = multi("Specialist", "owner_specialist", default_all=False)
f_geo = multi("Geography", "primary_geo", default_all=False)
f_acct = multi("Account", "account_name", default_all=False)
f_search = st.sidebar.text_input("Campaign name contains")

dim = df_all
for col, sel in [("campaign_type", f_type), ("campaign_status", f_status), ("owner_specialist", f_owner),
                 ("primary_geo", f_geo), ("account_name", f_acct)]:
    if sel:
        dim = dim[dim[col].astype(str).isin(sel)]
if f_search:
    dim = dim[dim["campaign_name"].str.contains(f_search, case=False, na=False)]

if dim.empty:
    st.warning("No campaigns match these filters.")
    st.stop()

# ---------------------------------------------------------------------------
# Core calculations
# ---------------------------------------------------------------------------
attrs = latest_attributes(dim)
window = dim[(dim["date"] >= w_start) & (dim["date"] <= w_end)]
n_days = (w_end - w_start).days + 1
prev = dim[(dim["date"] >= w_start - pd.Timedelta(days=n_days)) & (dim["date"] < w_start)]

pace, ctx = pacing(dim, targets)
camp = summarize(window).merge(attrs, on="campaign_id", how="left")
health = health_scores(camp, pace)
trend = period_compare(dim, w_end, n_days)
actions = campaign_actions(health, pace, trend)

tot = summarize(window.assign(_all=1), by="_all").iloc[0] if len(window) else None
tot_prev = summarize(prev.assign(_all=1), by="_all").iloc[0] if len(prev) else None

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("Google Ads Command Center")
cap = (f"Data: **{src}** · Targets: **{src_t}** · {dim['campaign_id'].nunique():,} campaigns · "
       f"Window {w_start:%d %b} – {w_end:%d %b %Y} · Data current to **{ctx['as_of']:%d %b %Y}**")
st.caption(cap)
if "SAMPLE" in src:
    st.warning("You are looking at anonymised SAMPLE data. Upload your export in the sidebar → Data sources.")
for w in warns:
    st.info(w)
if targets is None:
    st.warning("No campaign targets loaded: pacing shows 'No target' and CPA/ROAS rules are off. "
               "Upload campaign_targets.csv (sidebar → Data sources).")
if (pd.Timestamp.today().normalize() - ctx["as_of"]).days > 3:
    st.warning(f"Latest data is {ctx['as_of']:%d %b}. Pacing is calculated as of that date, not today.")

tabs = st.tabs(["🚨 Action Queue", "💰 Budget Pacing", "📈 Overview", "🩺 Campaign Health",
                "🔑 Keywords & Search Terms", "📝 Ads", "🌐 Landing Pages"])

# ---------------------------------------------------------------------------
# Action Queue
# ---------------------------------------------------------------------------
with tabs[0]:
    if actions.empty:
        st.success("No actions triggered for the current filters and window.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        for col, p in zip([c1, c2, c3, c4], ["Critical", "High", "Medium", "Low"]):
            sub = actions[actions["Priority"] == p]
            col.metric(f"{p}", f"{len(sub)}", help=f"Money at stake: {money(sub['Money at stake'].sum(), cur)}")
        crit_stake = actions.loc[actions["Priority"].isin(["Critical", "High"]), "Money at stake"].sum()
        st.markdown(f"**Critical + High money at stake: {money(crit_stake, cur)}** "
                    "— wasted spend, overspend, missed conversions, or capped profitable volume, by rule.")
        pick = st.multiselect("Priority", ["Critical", "High", "Medium", "Low"], default=["Critical", "High"])
        view = actions[actions["Priority"].isin(pick)] if pick else actions
        st.dataframe(view.drop(columns="campaign_id"), use_container_width=True, hide_index=True,
                     column_config={"Money at stake": NUM("Money at stake", format="%.0f")},
                     height=min(36 * (len(view) + 1) + 3, 700))
        st.download_button("Download action queue (CSV)", actions.to_csv(index=False).encode(),
                           f"action_queue_{ctx['as_of']:%Y%m%d}.csv", "text/csv")
    with st.expander("How the rules work"):
        st.markdown(f"""
| Priority | Rule | Guard against noise |
|---|---|---|
| Critical | Zero conversions | Spend ≥ {RULES['zero_conv_min_spend_x_cpa']:.0f}× target CPA (or account CPA if no target) |
| Critical | Pacing > {RULES['pace_critical']:.0f}% **and** CPA above target | Needs a target CPA |
| High | Pacing < {RULES['pace_low_high']:.0f}% **and** conversions behind pro-rated goal | Needs budget + target CPA |
| High | CPA > target by {RULES['cpa_over_target']:.0%}+ / ROAS below target | Needs a target |
| High | CPA up > {RULES['cpa_spike']:.0%} vs previous period | ≥{RULES['min_conv_for_trend']} conv in both periods and ≥{RULES['cpa_spike_z']:.0f}σ conversion drop |
| High | > {RULES['lost_is_budget_high']:.0%} IS lost to budget **and** at target efficiency | Inefficient ones go to Medium: don't scale them |
| Medium | CTR < {RULES['ctr_below_median']:.0%} of campaign-type median | ≥{RULES['min_impr_ctr']:,} impressions |
| Medium | CVR < {RULES['cvr_below_median']:.0%} of type median with healthy CTR | ≥{RULES['min_clicks_cvr']} clicks |
| Low | Strong ROAS with budget headroom; spend with no plan on file | — |

Thresholds live in `scripts/generate_recommendations.py → RULES`. Benchmarks are your own account's medians
by campaign type, not industry averages.
""")

# ---------------------------------------------------------------------------
# Budget Pacing
# ---------------------------------------------------------------------------
with tabs[1]:
    st.subheader(f"Month-to-date pacing — {ctx['as_of']:%B %Y}, day {ctx['days_elapsed']} of {ctx['days_in_month']}")
    if ctx["early_month"]:
        st.info(f"Only {ctx['days_elapsed']} days into the month. Pacing swings hard on 1–2 days of data; "
                "act on Critical items only until day 7.")
    planned = pace["planned_monthly_spend"].sum()
    spent = pace["spend"].sum()
    proj = pace["projected_month_end"].sum()
    exp = pace["expected_spend_to_date"].sum()
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("MTD spend", money(spent, cur))
    k2.metric("Expected to date", money(exp, cur), delta=f"{(spent / exp - 1):+.1%} vs plan" if exp else None,
              delta_color="off")
    k3.metric("Planned month", money(planned, cur))
    k4.metric("Projected month-end", money(proj, cur),
              delta=f"{(proj / planned - 1):+.1%} vs plan" if planned else None, delta_color="inverse")

    counts = pace["pacing_status"].value_counts()
    order = ["Overspending", "Underspending", "Zero spend", "No target", "On pace", "Inactive"]
    cc = st.columns(len(order))
    for col, s in zip(cc, order):
        col.metric(s, int(counts.get(s, 0)))

    status_pick = st.multiselect("Show status", order, default=[s for s in order[:4] if counts.get(s, 0)])
    pv = pace[pace["pacing_status"].isin(status_pick)] if status_pick else pace
    show(pv, {"campaign_name": "Campaign", "campaign_type": "Type", "owner_specialist": "Owner",
              "pacing_status": "Status", "pacing_pct": "Pacing %", "spend": "MTD spend",
              "expected_spend_to_date": "Expected", "planned_monthly_spend": "Plan",
              "run_rate_daily": "7-day run rate/day", "projected_month_end": "Projected",
              "projected_vs_plan_pct": "Proj. vs plan %", "daily_budget": "Daily budget",
              "conversions": "MTD conv.", "mtd_cpa": "MTD CPA", "target_cpa": "Target CPA",
              "mtd_roas": "MTD ROAS", "target_roas": "Target ROAS"},
         pct_cols=("pacing_pct", "projected_vs_plan_pct"), key="pace_tbl")

    plot = pace[pace["pacing_pct"].notna() & (pace["spend"] > 0)]
    if len(plot):
        fig = px.scatter(plot, x="planned_monthly_spend", y="pacing_pct", color="pacing_status",
                         hover_name="campaign_name", log_x=True, size="spend", size_max=28,
                         labels={"planned_monthly_spend": "Planned monthly spend (log)", "pacing_pct": "Pacing %"},
                         color_discrete_map={"Overspending": "#D62728", "Underspending": "#FF7F0E",
                                             "On pace": "#2CA02C"})
        fig.add_hrect(y0=90, y1=110, fillcolor="green", opacity=0.07, line_width=0)
        fig.update_layout(height=420, margin=dict(t=20, b=10))
        st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# Overview
# ---------------------------------------------------------------------------
with tabs[2]:
    if tot is None:
        st.info("No data in the selected window.")
    else:
        g = lambda s, k: s[k] if s is not None else np.nan  # noqa: E731
        m = st.columns(7)
        m[0].metric("Spend", money(tot["spend"], cur), delta(tot["spend"], g(tot_prev, "spend")), delta_color="off")
        m[1].metric("Conversions", f"{tot['conversions']:,.0f}", delta(tot["conversions"], g(tot_prev, "conversions")))
        m[2].metric("CPA", money(tot["cpa"], cur), delta(tot["cpa"], g(tot_prev, "cpa")), delta_color="inverse")
        m[3].metric("Conv. value", money(tot["conversion_value"], cur), delta(tot["conversion_value"], g(tot_prev, "conversion_value")))
        m[4].metric("ROAS", f"{tot['roas']:.2f}" if np.isfinite(tot["roas"]) else "n/a", delta(tot["roas"], g(tot_prev, "roas")))
        m[5].metric("CTR", f"{tot['ctr']:.2%}", delta(tot["ctr"], g(tot_prev, "ctr")))
        m[6].metric("Conv. rate", f"{tot['conv_rate']:.2%}", delta(tot["conv_rate"], g(tot_prev, "conv_rate")))
        st.caption(f"Deltas compare against the previous {n_days} days.")

        daily = summarize(window, by="date")
        metric = st.radio("Trend", ["spend", "conversions", "cpa", "roas", "ctr"], horizontal=True,
                          format_func=lambda x: {"spend": "Spend", "conversions": "Conversions", "cpa": "CPA",
                                                 "roas": "ROAS", "ctr": "CTR"}[x])
        fig = px.line(daily, x="date", y=metric, markers=True)
        fig.update_layout(height=320, margin=dict(t=10, b=10), yaxis_title=None, xaxis_title=None)
        st.plotly_chart(fig, use_container_width=True)

        c1, c2 = st.columns(2)
        by_type = summarize(window, by="campaign_type")
        by_type["spend_share"] = by_type["spend"] / by_type["spend"].sum()
        by_type["conv_share"] = by_type["conversions"] / max(by_type["conversions"].sum(), 1e-9)
        acct_roas = tot["roas"]
        by_type["efficiency_index"] = by_type["roas"] / acct_roas if np.isfinite(acct_roas) and acct_roas else np.nan
        with c1:
            st.markdown("**By campaign type**")
            show(by_type.sort_values("spend", ascending=False),
                 {"campaign_type": "Type", "spend": "Spend", "spend_share": "Spend share",
                  "conversions": "Conv.", "conv_share": "Conv. share", "cpa": "CPA", "roas": "ROAS",
                  "efficiency_index": "Efficiency idx"}, frac_cols=("spend_share", "conv_share"), key="bytype")
            st.caption("Efficiency index = type ROAS ÷ account ROAS. Spend share well above conv. share = budget leak.")
        with c2:
            st.markdown("**By specialist**")
            by_owner = summarize(window, by="owner_specialist")
            n_actions = actions.groupby("Owner").size() if not actions.empty else pd.Series(dtype=int)
            by_owner["open_actions"] = by_owner["owner_specialist"].map(n_actions).fillna(0)
            show(by_owner.sort_values("spend", ascending=False),
                 {"owner_specialist": "Specialist", "spend": "Spend", "conversions": "Conv.", "cpa": "CPA",
                  "roas": "ROAS", "open_actions": "Open actions"}, key="byowner")

# ---------------------------------------------------------------------------
# Campaign Health
# ---------------------------------------------------------------------------
with tabs[3]:
    bands = health["health_band"].value_counts()
    b1, b2, b3 = st.columns(3)
    b1.metric("Poor (<50)", int(bands.get("Poor", 0)))
    b2.metric("Watch (50–74)", int(bands.get("Watch", 0)))
    b3.metric("Healthy (75+)", int(bands.get("Healthy", 0)))
    hv = health.merge(trend[["campaign_id", "spend_chg_pct", "conversions_chg_pct", "cpa_chg_pct"]],
                      on="campaign_id", how="left")
    show(hv, {"campaign_name": "Campaign", "campaign_type": "Type", "owner_specialist": "Owner",
              "health_score": "Score", "health_band": "Band", "spend": "Spend", "conversions": "Conv.",
              "cpa": "CPA", "target_cpa": "Target CPA", "roas": "ROAS", "target_roas": "Target ROAS",
              "ctr": "CTR", "conv_rate": "CVR", "search_impression_share": "Search IS",
              "search_lost_is_budget": "Lost IS (budget)", "search_lost_is_rank": "Lost IS (rank)",
              "pacing_pct": "Pacing %", "spend_chg_pct": "Spend Δ%", "conversions_chg_pct": "Conv Δ%",
              "cpa_chg_pct": "CPA Δ%", "pen_efficiency": "−Efficiency", "pen_zero_conv": "−Zero conv",
              "pen_pacing": "−Pacing", "pen_budget_lost_is": "−Lost IS"},
         pct_cols=("pacing_pct", "spend_chg_pct", "conversions_chg_pct", "cpa_chg_pct"),
         frac_cols=("ctr", "conv_rate", "search_impression_share", "search_lost_is_budget", "search_lost_is_rank"),
         height=600, key="health_tbl")
    with st.expander("How the health score is calculated"):
        st.markdown("""
Starts at **100** and deducts, so every point is traceable in the `−` columns:
- **Efficiency (up to −40):** 1 point per 1% CPA above target, or per 1% ROAS below target.
- **Zero conversions on material spend (−30):** spend ≥ 2× account CPA with no conversions.
- **Pacing (up to −15):** 1 point per 2 percentage points outside 90–110%.
- **Budget-lost impression share (up to −15):** 0.5 point per point of Search IS lost to budget.

Campaigns without targets are scored only on what can be measured — set targets to make the score meaningful.
""")

# ---------------------------------------------------------------------------
# Keywords & search terms
# ---------------------------------------------------------------------------
acct_cpa = tot["cpa"] if tot is not None else np.nan
with tabs[4]:
    kw, kw_w, kw_src = load("keywords", up_kw, allow_sample=USING_SAMPLE)
    stt, st_w, st_src = load("search_terms", up_st, allow_sample=USING_SAMPLE)
    if kw is None and stt is None:
        st.info("Upload a keyword report and/or search terms report (sidebar → Data sources) to activate this page.")
    if kw is not None:
        st.caption(f"Keywords: {kw_src} · zero-conversion threshold = 2× account CPA ({money(2 * acct_cpa, cur)})")
        if "campaign_name" in kw.columns:
            kw = kw[kw["campaign_name"].isin(dim["campaign_name"].unique())] if f_type or f_owner or f_geo or f_acct or f_search else kw
        views = keyword_views(kw, acct_cpa)
        sel = st.radio("Keyword view", list(views), horizontal=True)
        show(views[sel], {"campaign_name": "Campaign", "ad_group": "Ad group", "keyword": "Keyword",
                          "match_type": "Match", "quality_score": "QS", "cost": "Cost", "impressions": "Impr.",
                          "clicks": "Clicks", "ctr": "CTR", "conversions": "Conv.", "conv_rate": "CVR",
                          "cpa": "CPA", "roas": "ROAS"}, frac_cols=("ctr", "conv_rate"), key="kw_tbl")
    if stt is not None:
        st.divider()
        st.caption(f"Search terms: {st_src} · negative-candidate threshold = 1× account CPA ({money(acct_cpa, cur)})")
        sv = search_term_views(stt, acct_cpa)
        sel2 = st.radio("Search term view", list(sv), horizontal=True)
        show(sv[sel2], {"search_term": "Search term", "campaign_name": "Campaign", "ad_group": "Ad group",
                        "cost": "Cost", "clicks": "Clicks", "conversions": "Conv.", "cpa": "CPA"}, key="st_tbl")
        st.download_button("Download negative candidates (CSV)",
                           sv["Negative keyword candidates"].to_csv(index=False).encode(),
                           "negative_keyword_candidates.csv", "text/csv")
        st.caption("Review before adding: a term with zero conversions in 30 days can still be a B2B lead with a long cycle.")

# ---------------------------------------------------------------------------
# Ads
# ---------------------------------------------------------------------------
with tabs[5]:
    ads, ad_w, ad_src = load("ads", up_ads, allow_sample=USING_SAMPLE)
    if ads is None:
        st.info("Upload an ad report (sidebar → Data sources) to activate this page.")
    else:
        min_clicks = st.slider("Minimum clicks per ad to compare", 30, 500, 100, 10,
                               help="Below ~100 clicks, conversion-rate differences between ads are mostly noise.")
        av = ad_views(ads, acct_cpa, min_clicks=min_clicks)
        sel3 = st.radio("Ad view", list(av), horizontal=True)
        show(av[sel3], {"campaign_name": "Campaign", "ad_group": "Ad group", "headline": "Headline / ad",
                        "ad_type": "Type", "cost": "Cost", "clicks": "Clicks", "ctr": "CTR",
                        "conversions": "Conv.", "conv_rate": "CVR", "cvr_vs_ad_group": "CVR vs ad group",
                        "cpa": "CPA"}, frac_cols=("ctr", "conv_rate", "cvr_vs_ad_group"), key="ad_tbl")
        st.caption("Ads are compared only against other ads in the same ad group with at least the minimum clicks.")

# ---------------------------------------------------------------------------
# Landing pages
# ---------------------------------------------------------------------------
with tabs[6]:
    lp = summarize(window, by="landing_page_url")
    lp["campaigns"] = lp["landing_page_url"].map(window.groupby("landing_page_url")["campaign_id"].nunique())
    lp["spend_share"] = lp["spend"] / lp["spend"].sum()
    acct_cvr = tot["conv_rate"] if tot is not None else np.nan
    lp["flag"] = np.select(
        [(lp["conversions"] == 0) & (lp["spend"] >= 2 * acct_cpa),
         (lp["clicks"] >= 200) & (lp["conv_rate"] < acct_cvr * 0.5),
         (lp["clicks"] >= 200) & (lp["conv_rate"] > acct_cvr * 1.5)],
        ["Spend, no conversions", "CVR < half account avg", "High converter — candidate for more budget"],
        default="")
    show(lp.sort_values("spend", ascending=False),
         {"landing_page_url": "Landing page", "flag": "Flag", "campaigns": "Campaigns", "spend": "Spend",
          "spend_share": "Spend share", "clicks": "Clicks", "conversions": "Conv.", "conv_rate": "CVR",
          "cpa": "CPA", "roas": "ROAS"}, frac_cols=("spend_share", "conv_rate"), key="lp_tbl")
    st.caption("Uses the Final URL from the campaign export. For page-level engagement, join GA4 landing-page data (phase 2).")
