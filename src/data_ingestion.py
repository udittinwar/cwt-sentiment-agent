"""Data ingestion helpers. Live social/news ingestion requires the user's API credentials."""
import os
import requests
import pandas as pd
import yfinance as yf
from datetime import datetime, timezone

def fetch_prices(ticker="BTC-USD", period="180d", interval="1h"):
    """Fetch public historical OHLCV data from Yahoo Finance."""
    df = yf.download(ticker, period=period, interval=interval, auto_adjust=True, progress=False)
    if df is None or df.empty:
        raise RuntimeError(f"No price data returned for {ticker}. Try interval='1d' or another ticker.")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.reset_index()
    time_col = "Datetime" if "Datetime" in df.columns else "Date"
    df = df.rename(columns={time_col: "timestamp", "Close": "close", "Open": "open",
                            "High": "high", "Low": "low", "Volume": "volume"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    for col in ["close", "open", "high", "low", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["timestamp", "close"]).sort_values("timestamp")

def fetch_tavily_news(query, max_results=8):
    key = os.getenv("TAVILY_API_KEY", "").strip()
    if not key:
        return [], "Tavily key not configured; news ingestion skipped."
    response = requests.post(
        "https://api.tavily.com/search",
        json={"api_key": key, "query": query, "topic": "news", "max_results": max_results,
              "search_depth": "basic", "include_answer": True},
        timeout=25
    )
    response.raise_for_status()
    data = response.json()
    return data.get("results", []), "Live Tavily results"

def fetch_apify_posts(query, max_items=200):
    token = os.getenv("APIFY_API_TOKEN", "").strip()
    actor = os.getenv("APIFY_ACTOR_ID", "").strip()
    if not token or not actor:
        return [], "Apify token/actor ID not configured; using demo sentiment data."
    # Generic actor input varies by actor. This payload suits common X scraper actors;
    # edit it to match the selected actor's input schema in Apify Console.
    url = f"https://api.apify.com/v2/acts/{actor}/run-sync-get-dataset-items"
    response = requests.post(url, params={"token": token, "clean": "true", "format": "json"},
                             json={"searchTerms": [query], "maxItems": max_items}, timeout=90)
    response.raise_for_status()
    items = response.json()
    posts = []
    for item in items:
        posts.append({
            "timestamp": item.get("createdAt") or item.get("created_at") or item.get("date"),
            "text": item.get("text") or item.get("fullText") or item.get("full_text") or "",
            "likes": item.get("likeCount", item.get("likes", 0)) or 0,
            "reposts": item.get("retweetCount", item.get("retweets", 0)) or 0,
            "views": item.get("viewCount", item.get("views", 0)) or 0,
            "followers": (item.get("author") or {}).get("followers", item.get("followers", 0)) or 0
        })
    return posts, f"Fetched {len(posts)} posts from configured Apify actor"

def demo_posts(start, end, asset="Bitcoin"):
    """Clearly labelled synthetic posts for pipeline demonstration—not real X data."""
    import numpy as np
    from datetime import timedelta
    rng = np.random.default_rng(42)
    phrases = [
        ("Bullish breakout looks possible", 0.8), ("Market feels strong today", 0.7),
        ("Price action is weak and risky", -0.7), ("Worried about another selloff", -0.8),
        ("Waiting for a clearer signal", 0.0), ("Long-term outlook remains positive", 0.6),
        ("Volatility is high; stay cautious", -0.3), ("No strong move yet", 0.0)
    ]
    times = pd.date_range(start=start, end=end, freq="2h", tz="UTC")
    rows = []
    for t in times:
        n = int(rng.integers(1, 5))
        for _ in range(n):
            text, _score = phrases[int(rng.integers(0, len(phrases)))]
            rows.append({"timestamp": t, "text": f"{asset}: {text}",
                         "likes": int(rng.integers(0, 500)),
                         "reposts": int(rng.integers(0, 100)),
                         "views": int(rng.integers(0, 15000)),
                         "followers": int(rng.integers(50, 100000))})
    return rows
