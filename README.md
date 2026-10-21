# CWT X Sentiment Price Prediction Agent

Python prototype for price forecasting with price and sentiment-derived features.

## Features
- Historical OHLCV ingestion through Yahoo Finance (`yfinance`)
- Optional X post ingestion through an Apify actor
- VADER sentiment scoring and time-bucket aggregation
- Optional latest-news search through Tavily
- Lagged price-return and sentiment features
- Random Forest forecasts for n+1 through n+5
- Chronological holdout and expanding-window walk-forward validation
- Price/sentiment chart, forecast table, and downloadable CSV
- Evaluation report template

## Important data disclosure
If API credentials are absent, the app creates **synthetic demo social posts** solely to demonstrate the pipeline. Those posts are not real X data and should never be represented as scraped data. The exact Apify request payload may need adapting to the selected actor's input schema.

## Setup (Windows / VS Code)
Open the project folder in VS Code Terminal:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
copy .env.example .env
streamlit run app.py
```

If PowerShell blocks activation, run:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## Optional API configuration
Edit `.env` and add:
- `APIFY_API_TOKEN`
- `APIFY_ACTOR_ID`
- `TAVILY_API_KEY`
- `EXA_API_KEY` (reserved for optional extension)

Apify actor schemas differ; confirm the actor input fields before relying on live ingestion. Do not commit `.env` or expose secrets in a public repository.

## Forecast interpretation
The app reports current price as the known n+0 reference and models future targets n+1 through n+5. In hourly mode, each horizon is one hour. In daily mode, each horizon is one day. Walk-forward validation uses chronological expanding training windows.

## Submission checklist
- [ ] Run the app and capture a screenshot of the dashboard
- [ ] Record actual output metrics from the run
- [ ] Update `reports/evaluation_report.md` with run-specific results
- [ ] Verify no API keys are committed
- [ ] Push to GitHub and include the repository URL in the submission email
- [ ] Disclose that live Apify/Tavily integrations were not run if no keys were available
