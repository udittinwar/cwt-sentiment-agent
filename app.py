import os, sys
from datetime import timedelta
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(__file__))
from src.data_ingestion import fetch_prices, fetch_apify_posts, fetch_tavily_news, demo_posts, load_local_apify_csv
from src.sentiment import score_posts, aggregate_sentiment
from src.feature_engineering import build_features
from src.model import train_forecast, walk_forward_evaluate
from src.evaluation import strategy_metrics

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
st.set_page_config(page_title="CWT X Sentiment Price Agent", layout="wide")
st.title("CrowdWisdomTrading — X Sentiment Price Prediction Agent")
st.caption("Prototype for research and evaluation. Predictions are experimental, not financial advice.")

with st.sidebar:
    st.header("Configuration")
    asset = st.selectbox("Asset", ["Bitcoin (BTC-USD)", "Ethereum (ETH-USD)", "Apple (AAPL)", "Tesla (TSLA)"])
    ticker = asset.split("(")[1].split(")")[0]
    interval_label = st.selectbox("Price interval", ["1 hour", "1 day"])
    interval = "1h" if interval_label == "1 hour" else "1d"
    period = st.selectbox("History", ["180 days", "60 days"], index=0)
    period_arg = "180d" if period == "180 days" else "60d"
    use_demo = st.checkbox("Use demo sentiment if Apify unavailable", value=True)
    refresh = st.button("Fetch data & run model", type="primary")

if "run" not in st.session_state: st.session_state.run = False
if refresh: st.session_state.run = True

if not st.session_state.run:
    st.info("Choose settings in the sidebar and click **Fetch data & run model**. Price data is fetched publicly; social posts use Apify if configured, otherwise clearly labelled synthetic demo posts.")
    st.markdown("**Pipeline:** ingestion → sentiment → features → model → walk-forward validation → evaluation.")
    st.stop()

try:
    with st.spinner("Fetching historical prices..."):
        prices = fetch_prices(ticker, period_arg, interval)
    start, end = prices["timestamp"].min(), prices["timestamp"].max()
    query = f'"{ticker.replace("-USD","")} OR {ticker}"'
    posts, post_status = load_local_apify_csv()
    sentiment_source = ""
    if posts:
        sentiment_source = "Real Apify X posts (local CSV; limited sample)"
    else:
        try:
            posts, post_status = fetch_apify_posts(query, max_items=20)
        except Exception as exc:
            post_status = f"Apify unavailable: {exc}"
            posts = []

    if posts and sentiment_source != "Real Apify X posts (local CSV; limited sample)":
        sentiment_source = "Live Apify X data"
    elif use_demo:
        posts = demo_posts(start, end, ticker)
        sentiment_source = "SYNTHETIC DEMO DATA — not real X posts"
        post_status += "; synthetic demo posts used instead"
    else:
        st.error("No X posts found. Enable demo mode or configure a working Apify actor.")
        st.stop()
    posts_df = score_posts(posts)
    sent = aggregate_sentiment(posts_df, interval=interval)
    df = build_features(prices, sent)
    st.caption(f"Price source: Yahoo Finance | Social source: {sentiment_source} | {post_status}")
    st.caption(f"Data window: {start} to {end} (UTC) | Rows: {len(df):,}")

    news, news_status = fetch_tavily_news(f"{ticker} latest financial news", max_results=8)
    with st.expander("Recent news (Tavily, if configured)", expanded=False):
        st.caption(news_status)
        if news:
            for item in news:
                st.markdown(f"**{item.get('title','Untitled')}** — {item.get('url','')}")
                st.write(item.get("content", "")[:500])
        else:
            st.write("No live news results available. Configure TAVILY_API_KEY to enable this integration.")

    latest = df.iloc[-1]
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Latest price", f"{latest['close']:,.4f}")
    c2.metric("Sentiment", f"{latest['sentiment_mean']:.3f}")
    c3.metric("Posts in latest bucket", f"{int(latest['post_count'])}")
    c4.metric("Data points", f"{len(df):,}")

    st.subheader("Price vs sentiment")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["close"], name="Asset price", yaxis="y"))
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["sentiment_mean"], name="Mean sentiment", yaxis="y2"))
    fig.update_layout(height=430, xaxis_title="Timestamp (UTC)", yaxis=dict(title="Price"),
                      yaxis2=dict(title="Sentiment (-1 to +1)", overlaying="y", side="right"),
                      legend=dict(orientation="h"))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Forecast horizons")
    forecast_rows = [{
        "Horizon": "n+0 (current observed price)",
        "Predicted price": float(df.iloc[-1]["close"]),
        "Test MAE": None,
        "Test RMSE": None,
    }]
    for h in range(1, 6):
        try:
            model, pred_df, metrics = train_forecast(df, h)
            forecast_rows.append({"Horizon": f"n+{h}", "Predicted price": float(model.predict(df.dropna(subset=[
                "close","sentiment_mean","post_count","positive_count","negative_count","engagement",
                "return_1","return_3","return_6","hour","day_of_week"])[[
                "close","sentiment_mean","post_count","positive_count","negative_count","engagement",
                "return_1","return_3","return_6","hour","day_of_week"]].tail(1))[0]),
                "Test MAE": metrics["MAE"], "Test RMSE": metrics["RMSE"]})
        except Exception as e:
            st.warning(f"n+{h} forecast unavailable: {e}")
    st.dataframe(pd.DataFrame(forecast_rows), use_container_width=True)

    st.subheader("Walk-forward validation")
    try:
        wf, wf_metrics = walk_forward_evaluate(df, horizon=1, step=24 if interval == "1h" else 1)
        st.write(wf_metrics)
        st.caption("Expanding-window evaluation. The strategy metrics below are illustrative and exclude fees/slippage.")
        wf_for_strategy = wf.copy()
        # Align close and target from the same timestamp rows used by the walk-forward output.
        aligned = df.merge(wf[["timestamp","prediction"]], on="timestamp", how="inner")
        aligned["target"] = aligned["target_n_plus_1"]
        strat = strategy_metrics(aligned.rename(columns={"prediction":"prediction"}).dropna(subset=["target"]))
        st.json(strat)
        wf_fig = go.Figure()
        wf_fig.add_trace(go.Scatter(x=wf["timestamp"], y=wf["actual"], name="Actual"))
        wf_fig.add_trace(go.Scatter(x=wf["timestamp"], y=wf["prediction"], name="Predicted"))
        wf_fig.update_layout(height=350, title="Walk-forward: n+1 actual vs predicted")
        st.plotly_chart(wf_fig, use_container_width=True)
    except Exception as e:
        st.warning(f"Walk-forward validation could not run: {e}")

    outlook = "Positive" if latest["sentiment_mean"] > 0.05 else ("Negative" if latest["sentiment_mean"] < -0.05 else "Neutral")
    st.subheader("Current sentiment outlook")
    st.write(f"**{ticker}: {outlook} sentiment** based on the latest aggregated sentiment score ({latest['sentiment_mean']:.3f}). "
             f"This summary uses {sentiment_source.lower()}. Synthetic demo sentiment must not be interpreted as actual market sentiment.")
    st.download_button("Download processed dataset CSV", df.to_csv(index=False).encode("utf-8"),
                       file_name=f"{ticker.lower()}_sentiment_features.csv", mime="text/csv")
except Exception as e:
    st.error(f"Pipeline error: {e}")
    st.info("Check internet connection and try again. If hourly data is unavailable for the selected asset, choose 1 day interval.")
