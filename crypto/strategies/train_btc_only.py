from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from xgboost import XGBClassifier

PREDICTION_THRESHOLD = 0.5


def load_btc_data() -> pd.DataFrame:
    path = Path("crypto/processed/BTCUSDT_1h_2020-05-01_2025-05-01_features.csv")
    return pd.read_csv(path, index_col=0, parse_dates=True)


def train_model(df: pd.DataFrame) -> None:
    features = ["rsi", "macd", "ema20", "ema50", "return", "volatility"]
    target = "target"

    split_point = int(len(df) * 0.8)
    train_df = df.iloc[:split_point]
    test_df = df.iloc[split_point:]

    x_train = train_df[features].astype(float)
    y_train = train_df[target].astype(int)
    x_test = test_df[features].astype(float)
    y_test = test_df[target].astype(int)

    # Модели
    xgb = XGBClassifier(use_label_encoder=False, eval_metric="logloss", verbosity=0)
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    lr = LogisticRegression(max_iter=1000)

    _ = xgb.fit(x_train, y_train)
    ensemble = VotingClassifier(estimators=[("xgb", xgb), ("rf", rf), ("lr", lr)], voting="hard")
    _ = ensemble.fit(x_train, y_train)

    probs = xgb.predict_proba(x_test)[:, 1]
    y_pred = (probs >= PREDICTION_THRESHOLD).astype(int)

    print("\n=== BTCUSDT — Ensemble Model Report ===")
    print(f"{classification_report(y_test, y_pred, digits=4)}")

    test_df = test_df.copy()
    test_df["prediction"] = y_pred
    test_df["prediction_prob"] = probs

    out_path = "crypto/processed/BTCUSDT_predictions.csv"
    test_df.to_csv(out_path)
    print(f"[✓] Saved to {out_path}")


if __name__ == "__main__":
    btc_df = load_btc_data()
    train_model(btc_df)
