import numpy as np
import pandas as pd

def strategy_metrics(predictions, annualization=365*24):
    """Illustrative long/short strategy: position follows predicted next-period return.
    Metrics are not investment advice; fees/slippage are not included.
    """
    df = predictions.copy()
    df["predicted_return"] = df["prediction"] / df["close"] - 1
    df["actual_return"] = df["target"] / df["close"] - 1
    df["position"] = np.sign(df["predicted_return"])
    df["strategy_return"] = df["position"] * df["actual_return"]
    r = df["strategy_return"].replace([np.inf, -np.inf], np.nan).dropna()
    if len(r) == 0:
        return {"Sharpe Ratio": np.nan, "Sortino Ratio": np.nan, "Max Drawdown": np.nan}
    std = r.std(ddof=1)
    downside = r[r < 0]
    downside_std = downside.std(ddof=1) if len(downside) > 1 else np.nan
    sharpe = (r.mean() / std * np.sqrt(annualization)) if std and not np.isnan(std) else np.nan
    sortino = (r.mean() / downside_std * np.sqrt(annualization)) if downside_std and not np.isnan(downside_std) else np.nan
    equity = (1 + r).cumprod()
    drawdown = equity / equity.cummax() - 1
    return {"Sharpe Ratio": float(sharpe), "Sortino Ratio": float(sortino),
            "Max Drawdown": float(drawdown.min())}
