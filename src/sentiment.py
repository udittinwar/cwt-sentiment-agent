import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def score_posts(posts):
    if not posts:
        return pd.DataFrame(columns=["timestamp", "text", "sentiment", "likes", "reposts", "views", "followers"])
    df = pd.DataFrame(posts)
    if "timestamp" not in df: df["timestamp"] = pd.NaT
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df["text"] = df.get("text", "").fillna("").astype(str) if isinstance(df.get("text", ""), pd.Series) else ""
    for c in ["likes", "reposts", "views", "followers"]:
        if c not in df: df[c] = 0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    analyzer = SentimentIntensityAnalyzer()
    df["sentiment"] = df["text"].apply(lambda x: analyzer.polarity_scores(x)["compound"])
    return df.dropna(subset=["timestamp"]).sort_values("timestamp")

def aggregate_sentiment(posts_df, interval="1h"):
    if posts_df.empty:
        return pd.DataFrame(columns=["timestamp", "sentiment_mean", "post_count", "positive_count", "negative_count", "engagement"])
    df = posts_df.copy().set_index("timestamp")
    agg = df.resample(interval).agg(
        sentiment_mean=("sentiment", "mean"),
        post_count=("text", "count"),
        positive_count=("sentiment", lambda s: int((s > 0.05).sum())),
        negative_count=("sentiment", lambda s: int((s < -0.05).sum())),
        engagement=("likes", "sum")
    ).fillna(0).reset_index()
    return agg
