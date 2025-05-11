import time
from pathlib import Path

import pandas as pd
import requests


def get_klines(symbol: str, interval: str, start_date: str, end_date: str) -> pd.DataFrame:
    base_url = "https://api.binance.com/api/v3/klines"
    start_ts = int(pd.to_datetime(start_date).timestamp() * 1000)
    end_ts = int(pd.to_datetime(end_date).timestamp() * 1000)
    all_data = []

    print(f"[+] Downloading {symbol} {interval} data from {start_date} to {end_date}...")

    while start_ts < end_ts:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": start_ts,
            "endTime": end_ts,
            "limit": 1000,
        }

        response = requests.get(base_url, params=params, timeout=30)
        data = response.json()

        if not data:
            break

        all_data += data
        start_ts = data[-1][6] + 1
        time.sleep(0.5)

    klines_df = pd.DataFrame(
        all_data,
        columns=[
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "quote_asset_volume",
            "number_of_trades",
            "taker_buy_base_volume",
            "taker_buy_quote_volume",
            "ignore",
        ],
    )

    klines_df["open_time"] = pd.to_datetime(klines_df["open_time"], unit="ms")
    klines_df = klines_df.set_index("open_time")
    klines_df = klines_df[["open", "high", "low", "close", "volume"]].astype(float)

    print(f"[✓] Retrieved {len(klines_df)} rows for {symbol} ({interval})")
    return klines_df


def save_to_csv(
    klines_df: pd.DataFrame, symbol: str, interval: str, start_date: str, end_date: str
) -> None:
    Path("crypto/data").mkdir(parents=True, exist_ok=True)
    filename = f"{symbol}_{interval}_{start_date}_{end_date}.csv"
    path = f"crypto/data/{filename}"
    klines_df.to_csv(path)
    print(f"[✓] Saved to {path}")


def download_symbols(symbols: list[str], interval: str, start_date: str, end_date: str) -> None:
    for symbol in symbols:
        klines_df = get_klines(symbol, interval, start_date, end_date)
        save_to_csv(klines_df, symbol, interval, start_date, end_date)


if __name__ == "__main__":
    download_symbols(
        symbols=["BTCUSDT", "ETHUSDT"],
        interval="1h",
        start_date="2020-05-01",
        end_date="2025-05-01",
    )
