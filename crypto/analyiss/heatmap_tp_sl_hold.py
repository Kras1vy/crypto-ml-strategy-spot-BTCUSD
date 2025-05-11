import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# === Параметры поиска ===
tp_range = np.arange(1.0, 3.1, 0.5)
sl_range = np.arange(0.5, 2.1, 0.5)
hold_range = [24]  # фиксируем HOLD для упрощения первого heatmap

# === Загрузка данных ===
price_data = pd.read_csv("crypto/processed/BTCUSDT_predictions.csv", parse_dates=True, index_col=0)


# === Функция: фильтрация по pred == 1 и расчет прибыли ===
def simulate_trades(
    price_df: pd.DataFrame, tp_factor: float, sl_factor: float, hold_hours: int
) -> float:
    equity = 1.0
    std = price_df["return"].rolling(window=24).std().fillna(0)

    # Iterate using integer index
    for idx in range(len(price_df)):
        row = price_df.iloc[idx]
        if row["prediction"] == 1:
            entry = row["close"]
            std_value = std.iloc[idx]  # Use iloc instead of loc with Hashable
            tp = entry * (1 + tp_factor * std_value)
            sl = entry * (1 - sl_factor * std_value)

            # Use integer indexing
            if idx + hold_hours >= len(price_df):
                continue

            future_window = price_df.iloc[idx : idx + hold_hours]

            for _, fut in future_window.iterrows():
                high = fut["high"]
                low = fut["low"]

                if high >= tp:
                    equity *= 1 + tp_factor * std_value
                    break

                if low <= sl:
                    equity *= 1 - sl_factor * std_value
                    break
            else:
                if idx + hold_hours - 1 < len(price_df):
                    equity *= price_df.iloc[idx + hold_hours - 1]["close"] / entry

    return equity - 1  # cumulative return


# === Сетка результатов ===
results = []

for tp in tp_range:
    for sl in sl_range:
        ret = simulate_trades(price_data.copy(), float(tp), float(sl), hold_range[0])
        results.append((tp, sl, ret))

# === Построение тепловой карты ===
results_df = pd.DataFrame(results, columns=["TP", "SL", "Return"])
# Use index and columns parameters for pivot
heatmap_data = pd.pivot_table(results_df, values="Return", index="SL", columns="TP").astype(float)

_ = plt.figure(figsize=(10, 6))
# Simplify heatmap call to avoid type issues
_ = sns.heatmap(heatmap_data, annot=True, fmt=".2%", cmap="RdYlGn", center=0)
_ = plt.title(f"Heatmap — Cumulative Return by TP/SL (Hold={hold_range[0]}h)")
_ = plt.xlabel("Take Profit Multiplier")
_ = plt.ylabel("Stop Loss Multiplier")
_ = plt.tight_layout()
_ = plt.show()
