from pathlib import Path

import pandas as pd

# Constants
TARGET_RETURN_THRESHOLD = 0.005  # 0.5% return threshold


def calculate_indicators(price_df: pd.DataFrame) -> pd.DataFrame:
    price_df = price_df.copy()

    # RSI
    delta = price_df["close"].astype(float).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    price_df["rsi"] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = price_df["close"].ewm(span=12, adjust=False).mean()
    ema26 = price_df["close"].ewm(span=26, adjust=False).mean()
    price_df["macd"] = ema12 - ema26

    # EMA
    price_df["ema20"] = price_df["close"].ewm(span=20, adjust=False).mean()
    price_df["ema50"] = price_df["close"].ewm(span=50, adjust=False).mean()

    # Return & Volatility
    price_df["return"] = price_df["close"].pct_change()
    price_df["volatility"] = price_df["close"].rolling(window=10).std()

    # 🎯 Новый таргет — движение вверх на ≥ 0.5% за 3 свечи (3 часа)
    future_return = (price_df["close"].shift(-3) - price_df["close"]) / price_df["close"]
    price_df["target"] = (future_return > TARGET_RETURN_THRESHOLD).astype(int)

    return price_df.dropna()


def process_all_crypto_data(
    data_dir: str = "crypto/data", output_dir: str = "crypto/processed"
) -> None:
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    for file_path in Path(data_dir).iterdir():
        if file_path.name.endswith(".csv"):
            print(f"[+] Processing {file_path.name}...")
            price_df = pd.read_csv(file_path, index_col=0, parse_dates=True)

            processed_df = calculate_indicators(price_df)

            out_name = file_path.name.replace(".csv", "_features.csv")
            out_path = Path(output_dir) / out_name
            processed_df.to_csv(out_path)

            print(f"[✓] Saved processed file to {out_path} — shape: {processed_df.shape}")


if __name__ == "__main__":
    process_all_crypto_data()
