import pandas as pd


def analyze_strategy_stats(csv_path: str) -> None:
    trade_data = pd.read_csv(csv_path)

    # Общее количество сделок
    total_trades = len(trade_data)

    # Доли по причинам выхода
    reason_counts = trade_data["reason"].value_counts(normalize=True)

    # Средняя длительность сделки (в часах)
    trade_data["entry_time"] = pd.to_datetime(trade_data["entry_time"])
    trade_data["exit_time"] = pd.to_datetime(trade_data["exit_time"])
    trade_data["duration_hours"] = (
        trade_data["exit_time"] - trade_data["entry_time"]
    ).dt.total_seconds() / 3600
    avg_duration = trade_data["duration_hours"].mean()

    # === Вывод результатов ===
    print("\n=== Strategy Stats ===")
    print(f"Total trades: {total_trades}")
    print("\nExit reason distribution:")
    for reason, pct in reason_counts.items():
        print(f"  {str(reason).upper()}: {pct:.2%}")
    print(f"\nAverage trade duration: {avg_duration:.2f} hours")


if __name__ == "__main__":
    analyze_strategy_stats("crypto/processed/BTCUSDT_backtest_results.csv")

