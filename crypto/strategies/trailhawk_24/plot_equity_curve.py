import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd


def plot_equity_curve(csv_path: str) -> None:
    equity_data = pd.read_csv(csv_path, parse_dates=["entry_time", "exit_time"])

    equity_data = equity_data.sort_values("exit_time")
    equity_data["equity"] = (1 + equity_data["pnl"]).cumprod()
    equity_data["drawdown"] = equity_data["equity"] / equity_data["equity"].cummax() - 1

    smoothed = equity_data["equity"].rolling(window=5, min_periods=1).mean()

    _, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12, 6), sharex=True, gridspec_kw={"height_ratios": [2, 1]}
    )

    ax1.plot(
        equity_data["exit_time"], equity_data["equity"], label="Equity", color="green", alpha=0.4
    )
    ax1.plot(equity_data["exit_time"], smoothed, label="Smoothed", color="black")
    ax1.set_title("BTCUSDT — Equity Curve")
    ax1.set_ylabel("Equity (x)")
    ax1.legend()

    ax2.plot(equity_data["exit_time"], equity_data["drawdown"], label="Drawdown", color="red")
    ax2.set_ylabel("Drawdown")
    ax2.set_title("Drawdown (from peak)")
    ax2.set_xlabel("Exit Time")

    ax2.xaxis.set_major_locator(mdates.MonthLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    _ = plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_equity_curve("crypto/processed/BTCUSDT_backtest_results.csv")

