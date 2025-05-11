import matplotlib.pyplot as plt
import pandas as pd

# Загрузка результатов
df = pd.read_csv("crypto/processed/BTCUSDT_backtest_results.csv", parse_dates=["exit_time"])

# Equity = cumulative_return + 1 (т.е. 0 → 1.0, +0.2 → 1.2)
df["equity"] = df["cumulative_return"] + 1.0

# Сглаженная equity (rolling window = 5)
df["equity_smooth"] = df["equity"].rolling(window=5).mean()

# Расчёт максимальной просадки
df["max_equity"] = df["equity"].cummax()
df["drawdown"] = (df["equity"] - df["max_equity"]) / df["max_equity"]

# Построение графика
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

# 📈 Equity curve
ax1.plot(df["exit_time"], df["equity"], label="Equity", color="green", alpha=0.4)
ax1.plot(df["exit_time"], df["equity_smooth"], label="Smoothed", color="black")
ax1.set_title("BTCUSDT — Equity Curve")
ax1.set_ylabel("Equity (x)")
ax1.legend()
ax1.grid(True)

# 📉 Drawdown
ax2.plot(df["exit_time"], df["drawdown"], label="Drawdown", color="red")
ax2.set_title("Drawdown (from peak)")
ax2.set_ylabel("Drawdown")
ax2.set_xlabel("Exit Time")
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()
