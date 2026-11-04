"""
Generate realistic, anonymised sample data so the dashboard runs before real exports exist.
Injects known problems (overpacing, zero-conversion spend, budget-limited winners, CPA spikes,
dead campaigns, weak landing pages) so every rule in the action queue can be verified.

    python scripts/generate_sample_data.py            # writes data/sample/*.csv
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)
OUT = Path(__file__).resolve().parents[1] / "data" / "sample"
END = pd.Timestamp("2026-11-03")
START = END - pd.Timedelta(days=95)
N_CAMPAIGNS = 300

TYPES = (["Search"] * 60 + ["Performance Max"] * 18 + ["Shopping"] * 8 + ["Demand Gen"] * 8 +
         ["Display"] * 4 + ["Video"] * 2)
PRODUCT_LINES = ["Connectors", "Cables", "Fiber Optics", "Antennas", "Backplane", "High-Speed I/O",
                 "Power", "RF", "Sensors", "Industrial", "Automotive", "Datacom"]
GEOS = ["India", "US", "Germany", "China", "Japan", "UK", "France", "Mexico"]
OWNERS = ["Specialist A", "Specialist B", "Specialist C", "Specialist D"]
PAGES = [f"https://www.example.com/products/{p.lower().replace(' ', '-')}" for p in PRODUCT_LINES]

PROFILE = {  # base CTR, CVR, CPC (INR), AOV
    "Search": (0.045, 0.040, 38, 6500), "Performance Max": (0.012, 0.030, 22, 7000),
    "Shopping": (0.010, 0.025, 15, 5200), "Demand Gen": (0.006, 0.010, 18, 6000),
    "Display": (0.004, 0.006, 8, 5000), "Video": (0.003, 0.004, 6, 5000),
}


def campaigns() -> pd.DataFrame:
    rows = []
    for i in range(N_CAMPAIGNS):
        ctype = TYPES[i % len(TYPES)]
        line = PRODUCT_LINES[i % len(PRODUCT_LINES)]
        geo = GEOS[(i // 3) % len(GEOS)]
        budget = float(RNG.choice([1500, 3000, 5000, 8000, 12000, 20000]))
        rows.append({
            "campaign_id": str(20_000_000_000 + i * 37),
            "campaign_name": f"{ctype} - {line} - {geo} - {i:03d}",
            "campaign_type": ctype, "campaign_status": "Enabled",
            "daily_budget": budget, "primary_geo": geo,
            "owner_specialist": OWNERS[i % len(OWNERS)],
            "landing_page_url": PAGES[i % len(PAGES)],
            "account_id": f"123-456-{7000 + i % 4}", "account_name": f"ACS {['APAC', 'Americas', 'EMEA', 'Global'][i % 4]}",
            "scenario": "normal",
        })
    c = pd.DataFrame(rows)
    # Inject scenarios on fixed indices so tests are deterministic
    c.loc[[0, 1, 2], "scenario"] = "overpace_inefficient"
    c.loc[[3, 4, 5, 6], "scenario"] = "zero_conv"
    c.loc[[7, 8, 9], "scenario"] = "budget_limited_winner"
    c.loc[[10, 11], "scenario"] = "cpa_spike"
    c.loc[[12, 13], "scenario"] = "dead"
    c.loc[[14, 15], "scenario"] = "weak_landing_page"
    c.loc[[16, 17], "scenario"] = "low_ctr"
    c.loc[[18], "scenario"] = "underpace"
    c.loc[c.index >= 290, "campaign_status"] = "Paused"
    c.loc[c.index >= 290, "scenario"] = "paused"
    return c


def daily(c: pd.DataFrame) -> pd.DataFrame:
    dates = pd.date_range(START, END, freq="D")
    out = []
    for _, r in c.iterrows():
        ctr, cvr, cpc, aov = PROFILE[r["campaign_type"]]
        q = RNG.uniform(0.7, 1.3)  # campaign quality multiplier
        for d in dates:
            util = RNG.uniform(0.88, 1.08)
            ctr_d, cvr_d, cpc_d = ctr * q, cvr * q, cpc * RNG.uniform(0.85, 1.15)
            lis_b = RNG.uniform(0.0, 0.08)
            sc = r["scenario"]
            if sc == "paused" and d > END - pd.Timedelta(days=40):
                continue
            if sc == "overpace_inefficient":
                util, cvr_d = 1.55, cvr_d * 0.45
            elif sc == "zero_conv":
                cvr_d = 0.0
            elif sc == "budget_limited_winner":
                util, cvr_d, lis_b = 1.0, cvr_d * 1.8, RNG.uniform(0.30, 0.45)
            elif sc == "cpa_spike" and d > END - pd.Timedelta(days=30):
                cvr_d *= 0.45
            elif sc == "dead" and d >= END.replace(day=1):
                continue
            elif sc == "weak_landing_page":
                ctr_d, cvr_d = ctr_d * 1.6, cvr_d * 0.2
            elif sc == "low_ctr":
                ctr_d = ctr_d * 0.25
            elif sc == "underpace":
                util = 0.35
            cost = r["daily_budget"] * util * RNG.uniform(0.9, 1.05)
            clicks = max(int(cost / cpc_d), 0)
            impressions = int(clicks / max(ctr_d, 1e-4))
            conv = float(RNG.binomial(clicks, min(cvr_d, 0.9))) if clicks else 0.0
            value = conv * aov * RNG.uniform(0.8, 1.2)
            is_search = r["campaign_type"] == "Search"
            lis_r = RNG.uniform(0.10, 0.35)
            out.append({
                "date": d.date().isoformat(), "account_id": r["account_id"], "account_name": r["account_name"],
                "campaign_id": r["campaign_id"], "campaign_name": r["campaign_name"],
                "campaign_type": r["campaign_type"], "campaign_status": r["campaign_status"],
                "daily_budget": r["daily_budget"], "currency": "INR",
                "impressions": impressions, "clicks": clicks, "cost": round(cost, 2),
                "conversions": conv, "conversion_value": round(value, 2),
                "search_impression_share": round(max(1 - lis_b - lis_r, 0.05), 4) if is_search else None,
                "search_lost_is_budget": round(lis_b, 4) if is_search else None,
                "search_lost_is_rank": round(lis_r, 4) if is_search else None,
                "landing_page_url": r["landing_page_url"], "primary_geo": r["primary_geo"],
                "owner_specialist": r["owner_specialist"],
            })
    return pd.DataFrame(out)


def targets(c: pd.DataFrame) -> pd.DataFrame:
    t = c[["campaign_id", "campaign_name", "owner_specialist"]].copy()
    t["planned_monthly_spend"] = (c["daily_budget"] * 30.4).round(-2)
    is_roas = c["campaign_type"].isin(["Performance Max", "Shopping"])
    t["target_cpa"] = np.where(is_roas, 0, (c["campaign_type"].map({k: v[2] / v[1] for k, v in PROFILE.items()}) * 1.30).round(-1))
    t["target_roas"] = np.where(is_roas, 4.0, 0)
    t.loc[c.index.isin([20, 21, 22]), "planned_monthly_spend"] = np.nan  # "no target" cases
    return t[["campaign_id", "campaign_name", "planned_monthly_spend", "target_cpa", "target_roas", "owner_specialist"]]


def keywords(c: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    terms = ["board to board connector", "high speed backplane connector", "fiber optic patch cord",
             "rf coaxial cable", "sma connector", "power connector 100a", "pcie gen 6 connector",
             "usb type c receptacle", "m12 connector", "qsfp cage", "free connector datasheet",
             "connector jobs", "how to crimp connector", "cheap cables", "ip67 circular connector"]
    match = ["Exact", "Phrase", "Broad"]
    search = c[c["campaign_type"] == "Search"].head(60)
    kw_rows, st_rows = [], []
    for _, r in search.iterrows():
        for j in range(6):
            term = terms[(int(r["campaign_id"]) // 37 + j) % len(terms)]
            junk = any(w in term for w in ["jobs", "free", "how to", "cheap"])
            clicks = int(RNG.integers(20, 600))
            cost = clicks * RNG.uniform(25, 60)
            conv = 0.0 if junk else float(RNG.binomial(clicks, 0.04))
            kw_rows.append({"Campaign": r["campaign_name"], "Ad group": f"AG {j % 3 + 1}", "Keyword": term,
                            "Match type": match[j % 3], "Impr.": int(clicks / RNG.uniform(0.01, 0.08)),
                            "Clicks": clicks, "Cost": round(cost, 2), "Conversions": conv,
                            "Conv. value": round(conv * 6500, 2), "Quality Score": int(RNG.integers(2, 10))})
            st_rows.append({"Search term": f"{term} {RNG.choice(['', 'price', 'supplier', 'india', 'datasheet'])}".strip(),
                            "Campaign": r["campaign_name"], "Ad group": f"AG {j % 3 + 1}",
                            "Added/Excluded": "None", "Impr.": int(clicks / 0.04), "Clicks": clicks,
                            "Cost": round(cost, 2), "Conversions": conv})
    return pd.DataFrame(kw_rows), pd.DataFrame(st_rows)


def ads(c: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in c[c["campaign_type"] == "Search"].head(60).iterrows():
        for ag in range(2):
            for v in range(3):
                clicks = int(RNG.integers(80, 900))
                cvr = 0.04 * RNG.uniform(0.3, 1.8)
                conv = float(RNG.binomial(clicks, cvr))
                rows.append({"Campaign": r["campaign_name"], "Ad group": f"AG {ag + 1}",
                             "Ad ID": f"{r['campaign_id']}{ag}{v}", "Ad type": "Responsive search ad",
                             "Headline 1": RNG.choice(["Engineered for 224G", "Ships in 48 Hours",
                                                       "Request a Sample", "Datasheets & CAD Models"]),
                             "Impr.": int(clicks / RNG.uniform(0.02, 0.07)), "Clicks": clicks,
                             "Cost": round(clicks * RNG.uniform(25, 55), 2), "Conversions": conv,
                             "Conv. value": round(conv * 6500, 2)})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    c = campaigns()
    daily(c).to_csv(OUT / "google_ads_campaign_daily.csv", index=False)
    targets(c).to_csv(OUT / "campaign_targets.csv", index=False)
    kw, st = keywords(c)
    kw.to_csv(OUT / "keywords.csv", index=False)
    st.to_csv(OUT / "search_terms.csv", index=False)
    ads(c).to_csv(OUT / "ads.csv", index=False)
    print("Sample data written to", OUT)
