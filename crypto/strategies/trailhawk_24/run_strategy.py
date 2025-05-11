import pandas as pd

# Constants
TRAIL_ACTIVATION_THRESHOLD = 0.012  # 1.2% price increase to activate trailing stop
RSI_OVERBOUGHT = 70  # RSI level considered overbought
COMMISSION = 0.0005  # 0.05% commission on entry and exit
START_BALANCE = 10_000.0  # Starting demo balance


def run_btc_backtest(
    path: str = "crypto/processed/BTCUSDT_predictions.csv",
    hold: int = 24,
    min_prob: float = 0.52,
    vol_threshold: float = 0.0007,
) -> None:
    price_data = pd.read_csv(path, index_col=0, parse_dates=True)
    trades = []
    balance = START_BALANCE

    price_data["volatility"] = price_data["close"].rolling(window=10).std()

    for i in range(len(price_data) - hold - 1):
        row = price_data.iloc[i]

        if (
            row["prediction"] != 1
            or row["prediction_prob"] < min_prob
            or row["ema20"] <= row["ema50"]
            or row["rsi"] >= RSI_OVERBOUGHT
            or row["volatility"] < vol_threshold
        ):
            continue

        entry_price_raw = row["close"]
        std = row["volatility"]
        tp = entry_price_raw + std * 1.5
        sl = entry_price_raw - std * 2

        # Пропуск сделок с низким потенциальным профитом
        expected_profit_ratio = (tp - entry_price_raw) / entry_price_raw
        if expected_profit_ratio < COMMISSION * 2:
            continue

        trail_active = False
        trail_stop = None
        reason = "timeout"
        j = hold

        for j in range(1, hold + 1):
            future_row = price_data.iloc[i + j]
            high = future_row["high"]
            low = future_row["low"]
            close = future_row["close"]

            if (
                not trail_active
                and (high - entry_price_raw) / entry_price_raw >= TRAIL_ACTIVATION_THRESHOLD
            ):
                trail_active = True
                trail_stop = close - std * 0.8

            if trail_active:
                new_trail = close - std * 0.8
                if trail_stop is None or new_trail > trail_stop:
                    trail_stop = new_trail
                if low <= trail_stop:
                    exit_price_raw = trail_stop
                    reason = "trail"
                    break

            if high >= tp:
                exit_price_raw = tp
                reason = "tp"
                break

            if low <= sl:
                exit_price_raw = sl
                reason = "sl"
                break
        else:
            exit_price_raw = price_data.iloc[i + hold]["close"]

        # Комиссия на вход и выход
        entry_price = entry_price_raw * (1 + COMMISSION)
        exit_price = exit_price_raw * (1 - COMMISSION)

        pnl_pct = (exit_price - entry_price) / entry_price
        trade_profit = balance * pnl_pct
        balance += trade_profit

        exit_index = price_data.index[i + j] if j <= hold else price_data.index[i + hold]

        trades.append(
            {
                "entry_time": price_data.index[i],
                "exit_time": exit_index,
                "entry_price": entry_price_raw,
                "exit_price": exit_price_raw,
                "entry_price_with_fee": entry_price,
                "exit_price_with_fee": exit_price,
                "pnl_pct": pnl_pct,
                "pnl_usd": trade_profit,
                "balance": balance,
                "reason": reason,
            }
        )

    trades_df = pd.DataFrame(trades)

    out_path = "crypto/processed/BTCUSDT_backtest_full_risk.csv"
    trades_df.to_csv(out_path)

    print("\n=== BTCUSDT — Backtest with Full Risk and Commission ===")
    print(f"Trades taken: {len(trades_df)}")
    print(f"Final balance: ${balance:.2f}")
    print(f"Total return: {(balance / START_BALANCE - 1):.2%}")
    print(f"Winrate: {(trades_df['pnl_pct'] > 0).mean():.2%}")


if __name__ == "__main__":
    run_btc_backtest()
