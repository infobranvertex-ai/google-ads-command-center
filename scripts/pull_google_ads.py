"""
PHASE 2 — pull campaign-level daily data straight from the Google Ads API.

Not needed for phase 1 (manual exports). Requires API access via an MCC with a developer token.
Credentials come ONLY from environment variables (GitHub Actions secrets / local .env), never files in the repo:

  GOOGLE_ADS_DEVELOPER_TOKEN, GOOGLE_ADS_CLIENT_ID, GOOGLE_ADS_CLIENT_SECRET,
  GOOGLE_ADS_REFRESH_TOKEN, GOOGLE_ADS_LOGIN_CUSTOMER_ID (MCC id, digits only),
  GOOGLE_ADS_CUSTOMER_IDS (comma-separated child account ids, digits only)

    pip install google-ads
    python scripts/pull_google_ads.py --days 95 --out data/raw/google_ads_campaign_daily.csv

Corrections vs the original build plan's GAQL:
  * date segment is segments.date (metrics.date does not exist)
  * budget is campaign_budget.amount_micros (campaign.daily_budget does not exist)
  * LAST_30_DAYS is too short: MTD pacing + previous-period comparison need ~95 days
  * final URL is not on the campaign resource; it comes from landing_page_view
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import date, timedelta

import pandas as pd

CAMPAIGN_QUERY = """
SELECT
  segments.date,
  customer.id, customer.descriptive_name, customer.currency_code,
  campaign.id, campaign.name, campaign.advertising_channel_type, campaign.status,
  campaign_budget.amount_micros,
  metrics.impressions, metrics.clicks, metrics.cost_micros,
  metrics.conversions, metrics.conversions_value,
  metrics.search_impression_share,
  metrics.search_budget_lost_impression_share,
  metrics.search_rank_lost_impression_share
FROM campaign
WHERE segments.date BETWEEN '{start}' AND '{end}'
  AND campaign.status != 'REMOVED'
"""

LANDING_QUERY = """
SELECT campaign.id, landing_page_view.unexpanded_final_url, metrics.cost_micros
FROM landing_page_view
WHERE segments.date BETWEEN '{start}' AND '{end}'
"""

TYPE_NAMES = {"SEARCH": "Search", "PERFORMANCE_MAX": "Performance Max", "SHOPPING": "Shopping",
              "DEMAND_GEN": "Demand Gen", "DISPLAY": "Display", "VIDEO": "Video",
              "MULTI_CHANNEL": "App", "LOCAL": "Local", "SMART": "Smart"}


def client():
    try:
        from google.ads.googleads.client import GoogleAdsClient
    except ImportError:
        sys.exit("pip install google-ads")
    need = ["GOOGLE_ADS_DEVELOPER_TOKEN", "GOOGLE_ADS_CLIENT_ID", "GOOGLE_ADS_CLIENT_SECRET",
            "GOOGLE_ADS_REFRESH_TOKEN", "GOOGLE_ADS_LOGIN_CUSTOMER_ID", "GOOGLE_ADS_CUSTOMER_IDS"]
    missing = [k for k in need if not os.environ.get(k)]
    if missing:
        sys.exit(f"Missing env vars: {', '.join(missing)}")
    return GoogleAdsClient.load_from_dict({
        "developer_token": os.environ["GOOGLE_ADS_DEVELOPER_TOKEN"],
        "client_id": os.environ["GOOGLE_ADS_CLIENT_ID"],
        "client_secret": os.environ["GOOGLE_ADS_CLIENT_SECRET"],
        "refresh_token": os.environ["GOOGLE_ADS_REFRESH_TOKEN"],
        "login_customer_id": os.environ["GOOGLE_ADS_LOGIN_CUSTOMER_ID"].replace("-", ""),
        "use_proto_plus": True,
    })


def pull(days: int) -> pd.DataFrame:
    gads = client()
    svc = gads.get_service("GoogleAdsService")
    end = date.today() - timedelta(days=1)  # yesterday: today's data is incomplete
    start = end - timedelta(days=days - 1)
    rows, lp_rows = [], []
    for cid in [c.strip().replace("-", "") for c in os.environ["GOOGLE_ADS_CUSTOMER_IDS"].split(",") if c.strip()]:
        for batch in svc.search_stream(customer_id=cid, query=CAMPAIGN_QUERY.format(start=start, end=end)):
            for r in batch.results:
                ctype = r.campaign.advertising_channel_type.name
                is_search = ctype == "SEARCH"
                rows.append({
                    "date": r.segments.date,
                    "account_id": str(r.customer.id), "account_name": r.customer.descriptive_name,
                    "currency": r.customer.currency_code,
                    "campaign_id": str(r.campaign.id), "campaign_name": r.campaign.name,
                    "campaign_type": TYPE_NAMES.get(ctype, ctype.title()),
                    "campaign_status": r.campaign.status.name.capitalize(),
                    "daily_budget": r.campaign_budget.amount_micros / 1e6,
                    "impressions": r.metrics.impressions, "clicks": r.metrics.clicks,
                    "cost": r.metrics.cost_micros / 1e6,
                    "conversions": r.metrics.conversions, "conversion_value": r.metrics.conversions_value,
                    "search_impression_share": r.metrics.search_impression_share if is_search else None,
                    "search_lost_is_budget": r.metrics.search_budget_lost_impression_share if is_search else None,
                    "search_lost_is_rank": r.metrics.search_rank_lost_impression_share if is_search else None,
                })
        for batch in svc.search_stream(customer_id=cid, query=LANDING_QUERY.format(start=start, end=end)):
            for r in batch.results:
                lp_rows.append({"campaign_id": str(r.campaign.id),
                                "landing_page_url": r.landing_page_view.unexpanded_final_url,
                                "cost": r.metrics.cost_micros / 1e6})
    df = pd.DataFrame(rows)
    if lp_rows and len(df):
        # Campaign-level file carries the campaign's highest-spend final URL
        lp = pd.DataFrame(lp_rows).groupby(["campaign_id", "landing_page_url"], as_index=False)["cost"].sum()
        top = lp.sort_values("cost").groupby("campaign_id").tail(1)[["campaign_id", "landing_page_url"]]
        df = df.merge(top, on="campaign_id", how="left")
    # owner_specialist / primary_geo / targets come from campaign_targets.csv, not the API
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=95)
    ap.add_argument("--out", default="data/raw/google_ads_campaign_daily.csv")
    a = ap.parse_args()
    out = pull(a.days)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    out.to_csv(a.out, index=False)
    print(f"Wrote {len(out):,} rows, {out['campaign_id'].nunique() if len(out) else 0:,} campaigns -> {a.out}")
