import pandas as pd

def build_features(prices, sentiment):
    p = prices.copy().sort_values("timestamp")
    s = sentiment.copy()
    if s.empty:
        s = pd.DataFrame(columns=["timestamp","sentiment_mean","post_count","positive_count","negative_count","engagement"])
    df = pd.merge_asof(p, s.sort_values("timestamp"), on="timestamp", direction="backward")
    for c in ["sentiment_mean","post_count","positive_count","negative_count","engagement"]:
        if c not in df: df[c] = 0
    df[["sentiment_mean","post_count","positive_count","negative_count","engagement"]] = df[
        ["sentiment_mean","post_count","positive_count","negative_count","engagement"]].fillna(0)
    df["return_1"] = df["close"].pct_change(1)
    df["return_3"] = df["close"].pct_change(3)
    df["return_6"] = df["close"].pct_change(6)
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    # Predict the price at t+h from information available at t.
    df["target_n_plus_0"] = df["close"]  # Current observed price, not a future forecast
    for h in range(1, 6):
        df[f"target_n_plus_{h}"] = df["close"].shift(-h)
    return df.replace([float("inf"), float("-inf")], float("nan"))
