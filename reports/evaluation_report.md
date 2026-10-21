# CWT X Sentiment Price Prediction Agent — Evaluation Report

## Data provenance and status
Historical price data is fetched through Yahoo Finance. News search uses Tavily if a valid key is configured. When Apify is missing or fails, the dashboard uses synthetic demo posts to test the pipeline. **Synthetic posts are not real X posts and must not be represented as scraped data.**

- Asset:
- Interval:
- Price data start/end (UTC):
- Number of price rows:
- Social data source (real Apify / synthetic demo):
- Number of real X posts, if applicable:
- Tavily status and article links:

## Forecasting objective
n+0 is the current observed price. n+1 through n+5 are future periods. In hourly mode each period is one hour; in daily mode each period is one day.

## Features and model
Features include current close, aggregated sentiment, post volume, positive/negative post counts, engagement, lagged returns (1, 3, 6 periods), hour and day of week. Model: Random Forest Regressor.

## Validation results
Fill in from the actual latest run. Do not invent or copy values from another run.

| Horizon | Test MAE | Test RMSE |
|---|---:|---:|
| n+1 | |
| n+2 | |
| n+3 | |
| n+4 | |
| n+5 | |

Walk-forward MAE:
Walk-forward RMSE:

## Strategy metrics
- Sharpe Ratio:
- Sortino Ratio:
- Maximum Drawdown:

These are illustrative results from a simple direction-following strategy and exclude fees, slippage, latency, funding, market impact and other real-world constraints. They are not proof of profitability.

## One-page current sentiment outlook
Summarize the latest score and its positive/neutral/negative classification. Clearly identify whether the posts are real X records or synthetic demo data. If synthetic, state that the result only demonstrates the pipeline and does not represent actual market sentiment. Summarize recent Tavily news separately with source links.

## Limitations
1. Real 180-day X post ingestion is not verified until a valid Apify actor successfully returns real posts.
2. Apify actor input schemas differ and may require actor-specific payload changes.
3. VADER may misread financial language, sarcasm, and ticker ambiguity.
4. Forecast accuracy does not imply trading profitability.
5. Free-tier quotas and data availability may change.
6. Strategy metrics exclude transaction costs.

## Reproducibility and security
Record asset, interval, data range, actor ID, validation settings and package versions. Never commit `.env` or API tokens to a public repository.
