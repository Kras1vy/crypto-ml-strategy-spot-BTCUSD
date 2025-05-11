# 🚀 Crypto Trading Strategy with Machine Learning (BTCUSDT 1H)

This project implements a backtestable crypto trading strategy for **BTCUSDT**, using **1-hour candles** and a combination of rule-based logic and ensemble machine learning filters.

> ✅ +30% return over 1 year (May 2024 – May 2025)
> 📉 Max drawdown: -11.92%
> 📊 Assets: BTCUSDT, Binance data
> 🔍 ML: XGBoost, Random Forest, Logistic Regression

---

## 🔍 Features

- Rule-based trading logic (RSI, EMA, volatility, returns)
- ML filtering using ensemble model (majority vote)
- Equity curve visualization, drawdown tracking, PnL distribution
- Fully offline backtesting engine
- Modular codebase ready for extension to futures or live trading

---

## 🧠 ML Model

The strategy uses an **ensemble of 3 classifiers** to predict high-probability entries:

- `XGBoostClassifier`
- `RandomForestClassifier`
- `LogisticRegression`

The models are trained on 4 years of historical data (2020–2024), and tested on the unseen out-of-sample period (2024–2025).

---

## 📂 Project Structure

```
crypto/
├── data/               <- Raw Binance data
├── processed/          <- Features, predictions, and results
├── strategies/
│   └── trailhawk_24/   <- Main strategy logic & ML model
├── analysis/           <- Performance plots, metrics, and heatmaps
```

---

## 📈 Backtest Results

- Period: **May 2024 – May 2025**
- Return: **+30.0%**
- Max Drawdown: **-11.92%**
- Exit breakdown:
  - Take Profit: 37.1%
  - Trailing Stop: 31.9%
  - Stop Loss: 26.6%
  - Timeout: 4.4%


---

## 💡 Future Work

- Futures version (with leverage and shorting)
- Integration with Bybit API for live trading
- Dynamic position sizing via model confidence
- Portfolio expansion across multiple assets

---

## 👤 Author

**Egor Gorin**
Machine Learning + Backend Developer
📍 Toronto
📧 Egorkagorin04@gmail.com
