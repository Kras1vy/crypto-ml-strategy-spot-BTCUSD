from typing import Any, cast

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Путь к файлу с результатами бэктеста
csv_path = "crypto/processed/BTCUSDT_backtest_full_risk.csv"

# Загружаем данные
backtest_data = pd.read_csv(csv_path, parse_dates=["entry_time", "exit_time"])

# === 1. Equity Curve ===
_ = plt.figure(figsize=(12, 5))
# Convert to lists with explicit types
x_values: list[Any] = cast("list[Any]", backtest_data["exit_time"].tolist())
y_values: list[float] = cast("list[float]", backtest_data["balance"].tolist())
_ = plt.plot(x_values, y_values, label="Equity")
_ = plt.title("Equity Curve")
_ = plt.xlabel("Date")
_ = plt.ylabel("Balance ($)")
_ = plt.grid(visible=True)
_ = plt.legend()
_ = plt.tight_layout()
_ = plt.show()

# === 2. Max Drawdown ===
backtest_data["peak"] = backtest_data["balance"].cummax()
backtest_data["drawdown"] = (backtest_data["balance"] - backtest_data["peak"]) / backtest_data[
    "peak"
]
max_dd = backtest_data["drawdown"].min()

print(f"\n📉 Max Drawdown: {max_dd:.2%}\n")

# === 3. Histogram of PnL % ===
_ = plt.figure(figsize=(8, 4))
_ = sns.histplot(data=backtest_data, x="pnl_pct", bins=50, kde=True, color="skyblue")
_ = plt.title("PnL Distribution (% per trade)")
_ = plt.xlabel("PnL %")
_ = plt.grid(visible=True)
_ = plt.tight_layout()
_ = plt.show()

# === 4. Pie Chart of Exit Reasons ===
reason_counts = backtest_data["reason"].value_counts()
# Create labels directly from strings
reason_labels: list[str] = [str(cast("Any", idx)) for idx in reason_counts.index]
_ = plt.figure(figsize=(6, 6))
_ = plt.pie(reason_counts, labels=reason_labels, autopct="%1.1f%%", startangle=140)
_ = plt.title("Exit Reasons Distribution")
_ = plt.tight_layout()
_ = plt.show()
