# Google Ads Command Center

A Streamlit dashboard that turns Google Ads exports for ~300 campaigns into a ranked list of
what to fix today: budget pacing, wasted spend, budget-capped winners, CPA spikes, weak ads,
keywords and landing pages.

Pages: **Action Queue** · **Budget Pacing** · **Overview** · **Campaign Health** ·
**Keywords & Search Terms** · **Ads** · **Landing Pages**

It runs immediately on anonymised sample data (300 campaigns, 95 days, with injected problems)
so you can check the logic before touching real data.

---

## 1. Run locally (10 minutes)

```bash
git clone <your-private-repo-url>
cd google-ads-command-center
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
pytest -q                                              # 7 tests should pass
streamlit run app/streamlit_app.py
```

To use real data locally without uploading every time, drop files here (gitignored):

| File | Put it at |
|---|---|
| Campaign daily export | `data/raw/google_ads_campaign_daily.csv` |
| Targets | `data/targets/campaign_targets.csv` (copy the TEMPLATE) |
| Keyword / search term / ad reports | `data/raw/keywords.csv`, `search_terms.csv`, `ads.csv` |

Sample keyword/ad/targets data is **never** mixed with real campaign data: once a real campaign
file is loaded, sample side-files are switched off.

---

## 2. Data you need

### 2.1 Campaign daily export (required)

In Google Ads → **Campaigns** → segment by **Day** → download **CSV**. Or build the same in
Reports. One row per campaign per day; 90+ days so pacing and previous-period comparisons work.

Required columns (Google UI header names are mapped automatically):

| Dashboard field | Google Ads UI header |
|---|---|
| date | Day |
| campaign_id | Campaign ID |
| campaign_name | Campaign |
| impressions | Impr. |
| clicks | Clicks |
| cost | Cost |
| conversions | Conversions |

Strongly recommended: Campaign type, Campaign state, Budget, Currency code, Conv. value,
Search impr. share, Search lost IS (budget), Search lost IS (rank), Final URL, Account, Customer ID.

The loader handles Google's report-title rows, the "Total" footer, `--`, currency symbols,
thousands separators, `< 10%` / `> 90%` impression-share buckets, UTF-16 "CSV (Excel)" downloads
and .xlsx. Ratio columns (CTR, CPC, CPA, ROAS) are **recomputed from base metrics**, never trusted
from the export, because averaging daily ratios gives the wrong answer.

### 2.2 Campaign targets (needed for pacing and efficiency rules)

`campaign_id, campaign_name, planned_monthly_spend, target_cpa, target_roas, owner_specialist, primary_geo`

* Use `0` or blank for "no target".
* `owner_specialist` and `primary_geo` here override the export (the API has neither field).
* Without this file, pacing shows "No target" everywhere and the CPA/ROAS rules stay off.
  **This file is the single biggest determinant of whether the dashboard is useful.**

### 2.3 Optional reports

* **Keywords:** Campaign, Ad group, Keyword, Match type, Impr., Clicks, Cost, Conversions, Conv. value, Quality Score
* **Search terms:** Search term, Campaign, Ad group, Added/Excluded, Clicks, Cost, Conversions
* **Ads:** Campaign, Ad group, Ad ID, Ad type, Headline 1, Impr., Clicks, Cost, Conversions, Conv. value

---

## 3. How the numbers are calculated

**Pacing (month-to-date, independent of the date filter)**

```
as_of            = latest date in the data (not today — exports lag)
expected_to_date = planned_monthly_spend × days_elapsed / days_in_month
pacing %         = MTD spend / expected_to_date × 100
projected        = MTD spend + (last-7-day daily run rate × remaining days)
```
Overspending > 110%, On pace 90–110%, Underspending < 90%, plus Zero spend / No target / Inactive.
In the first 6 days of a month, the page warns that pacing is noisy.

**Action Queue.** Rules and thresholds live in `scripts/generate_recommendations.py → RULES`.
Every rule has a materiality guard so 300 campaigns don't generate 300 alerts:

| Priority | Trigger | Guard |
|---|---|---|
| Critical | Zero conversions | spend ≥ 2× target CPA (or account CPA) |
| Critical | Pacing > 130% and CPA above target | target CPA required |
| High | Pacing < 70% and behind pro-rated conversion goal | budget + target CPA required |
| High | CPA > target +20% / ROAS < target | target required |
| High | CPA up > 20% vs previous period | ≥20 conv in both periods and a ≥3σ conversion drop |
| High | > 20% Search IS lost to budget, at target efficiency | inefficient ones become Medium: "fix before scaling" |
| Medium | CTR < 50% of the campaign-type median | ≥1,000 impressions |
| Medium | Good CTR, CVR < 50% of type median | ≥100 clicks |
| Low | Strong ROAS with budget headroom; spend with no plan | — |

Benchmarks are **your own account's medians by campaign type**, not industry averages.
Actions are sorted by priority, then **money at stake**, the rule's estimate of wasted spend,
overspend, or capped profitable volume in currency.

**Health score (0–100).** Starts at 100 and deducts for efficiency vs target (max −40),
zero conversions on material spend (−30), pacing outside 90–110% (max −15) and budget-lost
impression share (max −15). The deductions are shown as columns, so every score is traceable.

**Currencies are never summed.** If the export contains more than one currency, the sidebar
forces you to view one at a time.

---

## 4. Deploy to Streamlit Community Cloud

1. Create a **private** GitHub repo `google-ads-command-center` and push this folder.
   Check `git status` before the first commit: only `data/sample/` and the targets TEMPLATE
   should be in `data/`.
2. Go to share.streamlit.io → sign in with GitHub → **Create app** → pick the repo,
   branch `main`, main file `app/streamlit_app.py`, Python 3.11.
3. In the app's **Settings → Sharing**, restrict viewers to named email addresses.
4. Open the app → sidebar → **Data sources** → upload the export and the targets file.
   Uploads live in that browser session only. They are not written to the repo.

> **Before step 4 with real Amphenol data, get IT/InfoSec sign-off.** Uploaded files are processed
> on Streamlit's (Snowflake's) cloud servers. If that is not approved, run it locally (section 1)
> or on an internal server. The code is identical either way.

---

## 5. Phase 2: API automation

`scripts/pull_google_ads.py` pulls the same campaign-daily file from the Google Ads API.
It uses the corrected GAQL (`segments.date`, `campaign_budget.amount_micros`, 95 days, final URL
from `landing_page_view`). `.github/workflows/daily_refresh.yml` runs it on demand.

Prerequisites: an MCC with an approved **developer token**, OAuth client + refresh token, and
these GitHub repo secrets: `GOOGLE_ADS_DEVELOPER_TOKEN`, `GOOGLE_ADS_CLIENT_ID`,
`GOOGLE_ADS_CLIENT_SECRET`, `GOOGLE_ADS_REFRESH_TOKEN`, `GOOGLE_ADS_LOGIN_CUSTOMER_ID`,
`GOOGLE_ADS_CUSTOMER_IDS`.

The workflow deliberately does **not** commit data back to Git (Git history keeps it forever).
It uploads a 7-day artifact. Swap that step for an approved store (restricted Google Sheet,
GCS, BigQuery) and point the app at it.

---

## 6. Repository layout

```
app/streamlit_app.py              dashboard
scripts/process_data.py           export loader, column mapping, cleaning (also a CLI)
scripts/calculate_pacing.py       pacing, period comparison, health score
scripts/generate_recommendations.py  action-queue rules, keyword/search-term/ad views
scripts/generate_sample_data.py   regenerates data/sample/
scripts/pull_google_ads.py        phase 2 API pull
tests/test_logic.py               regression tests (pytest)
data/sample/                      anonymised demo data (committed)
data/raw/, data/processed/        real data, gitignored
data/targets/                     targets; only the TEMPLATE is committed
.github/workflows/ci.yml          runs tests on every push
.github/workflows/daily_refresh.yml  phase 2, manual trigger
```
