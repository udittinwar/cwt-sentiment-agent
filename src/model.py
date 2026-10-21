import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

FEATURES = ["close", "sentiment_mean", "post_count", "positive_count", "negative_count",
            "engagement", "return_1", "return_3", "return_6", "hour", "day_of_week"]

def train_forecast(df, horizon=1, test_fraction=0.2):
    target = f"target_n_plus_{horizon}"
    data = df.dropna(subset=FEATURES + [target]).copy()
    if len(data) < 100:
        raise ValueError(f"Not enough usable rows to train reliably (only {len(data)}). Try daily interval or another asset.")
    split = int(len(data) * (1 - test_fraction))
    train, test = data.iloc[:split], data.iloc[split:]
    model = RandomForestRegressor(n_estimators=120, max_depth=8, min_samples_leaf=3,
                                  random_state=42, n_jobs=-1)
    model.fit(train[FEATURES], train[target])
    pred = model.predict(test[FEATURES])
    rmse = float(np.sqrt(mean_squared_error(test[target], pred)))
    mae = float(mean_absolute_error(test[target], pred))
    out = test[["timestamp", "close", target]].copy()
    out["prediction"] = pred
    return model, out, {"MAE": mae, "RMSE": rmse, "train_rows": len(train), "test_rows": len(test)}

def walk_forward_evaluate(df, horizon=1, min_train_fraction=0.6, step=24):
    """Expanding-window walk-forward evaluation; refit at each test block."""
    target = f"target_n_plus_{horizon}"
    data = df.dropna(subset=FEATURES + [target]).reset_index(drop=True)
    if len(data) < 150:
        raise ValueError(f"Need more rows for walk-forward validation; available: {len(data)}")
    initial = max(100, int(len(data) * min_train_fraction))
    actuals, preds, times = [], [], []
    for start in range(initial, len(data), step):
        end = min(start + step, len(data))
        train, test = data.iloc[:start], data.iloc[start:end]
        model = RandomForestRegressor(n_estimators=80, max_depth=8, min_samples_leaf=3,
                                      random_state=42, n_jobs=-1)
        model.fit(train[FEATURES], train[target])
        pred = model.predict(test[FEATURES])
        actuals.extend(test[target].tolist())
        preds.extend(pred.tolist())
        times.extend(test["timestamp"].tolist())
    if not actuals:
        raise ValueError("Walk-forward produced no test predictions.")
    return pd.DataFrame({"timestamp": times, "actual": actuals, "prediction": preds}), {
        "MAE": float(mean_absolute_error(actuals, preds)),
        "RMSE": float(np.sqrt(mean_squared_error(actuals, preds)))
    }
