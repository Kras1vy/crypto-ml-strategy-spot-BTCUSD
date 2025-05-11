from pathlib import Path

import pandas as pd

# Constants
RSI_OVERBOUGHT = 75  # RSI level considered overbought
TRAIL_START_CANDLE = 2  # Candle number when trailing stop begins


def run_backtest(
    path: str,
    symbol: str,
    hold: int = 3,  # max hold time
    min_prob: float = 0.55,  # min ML confidence
    vol_threshold: float = 0.001,  # minimum volatility to even enter
) -> None:
    price_data = pd.read_csv(path, index_col=0, parse_dates=True)
    trades = []

    price_data["volatility"] = price_data["close"].rolling(window=10).std()
    price_data["volume_ma"] = price_data["volume"].rolling(window=20).mean()

    for i in range(len(price_data) - hold - 1):
        row = price_data.iloc[i]

        # ✅ ENTRY CONDITIONS (Smart Filter)
        if (
            row["prediction"] != 1
            or row["prediction_prob"] < min_prob
            or row["rsi"] >= RSI_OVERBOUGHT
            or row["ema20"] <= row["ema50"]
            or row["volatility"] < vol_threshold
        ):
            continue

        entry_price = row["close"]
        stop_loss = entry_price - row["volatility"] * 1.5
        take_profit = entry_price + row["volatility"] * 2.5
        trail_price = None
        exit_index = None
        reason = "hold"
        exit_price = None  # Initialize exit_price

        for j in range(1, hold + 1):
            future_row = price_data.iloc[i + j]
            high = future_row["high"]
            low = future_row["low"]

            # 📈 Take profit hit
            if high >= take_profit:
                exit_price = take_profit
                exit_index = price_data.index[i + j]
                reason = "tp"
                break

            # 🛑 Stop loss hit
            if low <= stop_loss:
                exit_price = stop_loss
                exit_index = price_data.index[i + j]
                reason = "sl"
                break

            # 📉 Trailing stop logic (optional)
            if j >= TRAIL_START_CANDLE:
                # if price moved in our direction, move up the stop
                new_trail = future_row["close"] - row["volatility"]
                if trail_price is None or new_trail > trail_price:
                    trail_price = new_trail
                if low < trail_price:
                    exit_price = trail_price
                    exit_index = price_data.index[i + j]
                    reason = "trail"
                    break

        # ⏱ No exit? Use last close
        if exit_index is None:
            exit_price = price_data.iloc[i + hold]["close"]
            exit_index = price_data.index[i + hold]
            reason = "timeout"

        pnl = (exit_price - entry_price) / entry_price
        trades.append(
            {
                "entry_time": price_data.index[i],
                "exit_time": exit_index,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "pnl": pnl,
                "reason": reason,
            }
        )

    trades_df = pd.DataFrame(trades)
    trades_df["cumulative_return"] = (1 + trades_df["pnl"]).cumprod() - 1

    # 💾 Save result
    out_path = f"crypto/processed/{symbol}_backtest_results.csv"
    trades_df.to_csv(out_path)

    print(f"\n=== {symbol} — Backtest Results ===")
    print(f"Trades taken: {len(trades_df)}")
    print(f"Final cumulative return: {trades_df['cumulative_return'].iloc[-1]:.2%}")
    print(f"Winrate: {(trades_df['pnl'] > 0).mean():.2%}")


def run_all_backtests() -> None:
    processed_dir = Path("crypto/processed")

    for file_path in processed_dir.iterdir():
        if file_path.name.endswith("_predictions.csv"):
            symbol = file_path.name.split("_")[0]
            run_backtest(str(file_path), symbol)


if __name__ == "__main__":
    run_all_backtests()
