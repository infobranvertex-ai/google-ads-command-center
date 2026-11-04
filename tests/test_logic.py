"""Run: pytest -q"""
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.calculate_pacing import health_scores, latest_attributes, pacing, summarize  # noqa: E402
from scripts.generate_recommendations import campaign_actions  # noqa: E402
from scripts.process_data import CAMPAIGN_REQUIRED, normalize, normalize_targets, read_any  # noqa: E402


def _daily(rows):
    base = dict(account_id="1", account_name="A", campaign_type="Search", campaign_status="Enabled",
                daily_budget=100, currency="INR", impressions=1000, clicks=50, conversions=1.0,
                conversion_value=500.0, landing_page_url="/x", primary_geo="IN", owner_specialist="S")
    df = pd.DataFrame([{**base, **r} for r in rows])
    out, _ = normalize(df, CAMPAIGN_REQUIRED)
    return out


def test_google_ui_export_is_mapped():
    csv = (
        "Campaign performance report\n"
        "1 October 2026 - 31 October 2026\n"
        "Day,Campaign ID,Campaign,Campaign type,Campaign state,Currency code,Impr.,Clicks,Cost,Conversions,"
        "Conv. value,Search impr. share,Search lost IS (budget),CTR\n"
        '2026-10-01,111,Search - A,Search,Enabled,INR,"1,200",60,"₹2,400.50",3,9000,< 10%,45.20%,5.00%\n'
        "2026-10-02,111,Search - A,Search,Enabled,INR,1000,40,1500,--,0,> 90%,0%,4%\n"
        "Total: Account,,,,,,2200,100,3900.5,3,9000,,,\n"
    )
    buf = io.BytesIO(csv.encode())
    buf.name = "export.csv"
    df, warns = normalize(read_any(buf), CAMPAIGN_REQUIRED)
    assert len(df) == 2  # title rows skipped, Total row dropped
    assert df["cost"].tolist() == pytest.approx([2400.50, 1500.0])
    assert df["impressions"].iloc[0] == 1200
    assert df["conversions"].iloc[1] == 0  # "--" -> 0
    assert df["search_impression_share"].tolist() == pytest.approx([0.05, 0.95])
    assert df["search_lost_is_budget"].iloc[0] == pytest.approx(0.452)
    assert "ctr" not in df.columns  # ratios are recomputed, never trusted
    assert any("bucketed" in w for w in warns)


def test_targets_merge_has_no_suffix_collision():
    """Original plan: targets repeated campaign_name -> campaign_name_x -> KeyError in action queue."""
    df = _daily([{"date": "2026-11-01", "campaign_id": "1", "campaign_name": "C1", "cost": 100}])
    t = normalize_targets(pd.DataFrame({"campaign_id": [1], "campaign_name": ["C1 old name"],
                                        "planned_monthly_spend": [3000], "target_cpa": [0],
                                        "owner_specialist": ["New Owner"]}))
    p, _ = pacing(df, t)
    assert "campaign_name" in p.columns and not any(c.endswith(("_x", "_y")) for c in p.columns)
    assert p.loc[0, "owner_specialist"] == "New Owner"
    assert np.isnan(p.loc[0, "target_cpa"])  # 0 in template = no target


def test_pacing_uses_mtd_and_data_date_not_window():
    """Previous-month spend must not inflate pacing; as-of = last data date."""
    rows = [{"date": f"2026-10-{d:02d}", "campaign_id": "1", "campaign_name": "C", "cost": 500} for d in range(1, 32)]
    rows += [{"date": f"2026-11-{d:02d}", "campaign_id": "1", "campaign_name": "C", "cost": 100} for d in range(1, 11)]
    df = _daily(rows)
    t = pd.DataFrame({"campaign_id": ["1"], "planned_monthly_spend": [3000.0]})
    p, ctx = pacing(df, t)
    assert ctx["days_elapsed"] == 10
    assert p.loc[0, "spend"] == pytest.approx(1000)
    assert p.loc[0, "pacing_pct"] == pytest.approx(100.0)
    assert p.loc[0, "projected_month_end"] == pytest.approx(3000.0)
    assert p.loc[0, "pacing_status"] == "On pace"


def test_projection_uses_recent_run_rate():
    rows = [{"date": f"2026-11-{d:02d}", "campaign_id": "1", "campaign_name": "C",
             "cost": 50 if d <= 13 else 200} for d in range(1, 21)]
    p, _ = pacing(_daily(rows), pd.DataFrame({"campaign_id": ["1"], "planned_monthly_spend": [3000.0]}))
    assert p.loc[0, "run_rate_daily"] == pytest.approx(200)  # last 7 days
    assert p.loc[0, "projected_month_end"] == pytest.approx(13 * 50 + 7 * 200 + 200 * 10)


def test_zero_conversion_flag_needs_material_spend():
    rows = [{"date": "2026-11-01", "campaign_id": "big", "campaign_name": "Big", "cost": 5000, "conversions": 0},
            {"date": "2026-11-01", "campaign_id": "tiny", "campaign_name": "Tiny", "cost": 40, "conversions": 0},
            {"date": "2026-11-01", "campaign_id": "ok", "campaign_name": "OK", "cost": 1000, "conversions": 10}]
    df = _daily(rows)
    t = pd.DataFrame({"campaign_id": ["big", "tiny", "ok"], "planned_monthly_spend": [1e5] * 3,
                      "target_cpa": [100.0] * 3})
    p, _ = pacing(df, t)
    h = health_scores(summarize(df).merge(latest_attributes(df), on="campaign_id"), p)
    a = campaign_actions(h, p)
    zero = a[a["Issue"] == "Material spend, zero conversions"]["Campaign"].tolist()
    assert zero == ["Big"]


def test_budget_limited_inefficient_is_not_told_to_scale():
    rows = [{"date": "2026-11-01", "campaign_id": "1", "campaign_name": "Bad", "cost": 1000, "conversions": 2,
             "search_lost_is_budget": 0.4}]
    df = _daily(rows)
    t = pd.DataFrame({"campaign_id": ["1"], "planned_monthly_spend": [30000.0], "target_cpa": [100.0]})
    p, _ = pacing(df, t)
    h = health_scores(summarize(df).merge(latest_attributes(df), on="campaign_id"), p)
    a = campaign_actions(h, p)
    assert "Efficient but budget-limited" not in a["Issue"].tolist()
    assert "Budget-limited but not efficient" in a["Issue"].tolist()


def test_sample_data_end_to_end():
    sample = ROOT / "data" / "sample" / "google_ads_campaign_daily.csv"
    if not sample.exists():
        pytest.skip("run scripts/generate_sample_data.py first")
    df, _ = normalize(read_any(sample), CAMPAIGN_REQUIRED)
    t = normalize_targets(pd.read_csv(ROOT / "data" / "sample" / "campaign_targets.csv"))
    p, _ = pacing(df, t)
    end = df["date"].max()
    win = df[df["date"] > end - pd.Timedelta(days=30)]
    h = health_scores(summarize(win).merge(latest_attributes(df), on="campaign_id"), p)
    a = campaign_actions(h, p)
    crit = set(a.loc[a["Priority"] == "Critical", "Campaign"].str[-3:])
    assert {"000", "001", "002", "003", "004", "005", "006"} <= crit  # injected scenarios caught
    assert len(a) < 0.25 * df["campaign_id"].nunique()  # queue stays actionable
