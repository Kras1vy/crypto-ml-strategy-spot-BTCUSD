from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from xgboost import XGBClassifier

# Constants
PREDICTION_THRESHOLD = 0.5  # Threshold for positive class prediction


def load_data(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, index_col=0, parse_dates=True)


def train_ensemble_model(df: pd.DataFrame, symbol: str) -> None:
    features = ["rsi", "macd", "ema20", "ema50", "return", "volatility"]
    target = "target"

    # 🧠 Split по времени — честный трейдерский подход
    split_point = int(len(df) * 0.8)
    train_df = df.iloc[:split_point]
    test_df = df.iloc[split_point:]

    # Extract features and target with explicit types
    x_train, y_train = train_df[features].astype(float), train_df[target].astype(int)
    x_test, y_test = test_df[features].astype(float), test_df[target].astype(int)

    # 📦 Модели
    xgb = XGBClassifier(use_label_encoder=False, eval_metric="logloss", verbosity=0)
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    lr = LogisticRegression(max_iter=1000)

    ensemble = VotingClassifier(estimators=[("xgb", xgb), ("rf", rf), ("lr", lr)], voting="hard")

    _ = xgb.fit(x_train, y_train)
    _ = ensemble.fit(x_train, y_train)
    probs = xgb.predict_proba(x_test)[:, 1]  # вероятность класса 1
    y_pred = (probs >= PREDICTION_THRESHOLD).astype(int)

    print(f"\n=== {symbol} — Ensemble Model Report ===")
    # Print classification report directly
    print(f"{classification_report(y_test, y_pred, digits=4)}")

    # 💾 Сохраняем test с prediction — для трейдера/бэктеста
    test_df = test_df.copy()
    test_df["prediction"] = y_pred
    test_df["prediction_prob"] = probs

    out_path = f"crypto/processed/{symbol}_predictions.csv"
    test_df.to_csv(out_path)
    print(f"[✓] Saved predictions to {out_path}")


def run_for_all() -> None:
    processed_dir = Path("crypto/processed")

    for file_path in processed_dir.iterdir():
        if file_path.name.endswith("_features.csv"):
            symbol = file_path.name.split("_")[0]
            crypto_df = load_data(file_path)
            train_ensemble_model(crypto_df, symbol)


if __name__ == "__main__":
    run_for_all()
