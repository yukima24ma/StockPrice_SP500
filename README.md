# S&P 500 Direction Prediction — ML Model Comparison

A machine learning portfolio project that compares seven classification models for predicting the **daily direction** of the S&P 500 index across three short-term horizons: **1 day**, **7 days**, and **30 days**.

> **Disclaimer:** This is an educational and portfolio project. Results should not be used for real trading decisions.

---

## Research Question

Can historical S&P 500 price patterns and technical indicators predict whether the index will go up or down over the next 1, 7, or 30 trading days?

---

## Project Structure

```
trading_model/
│
├── notebooks/
│   └── sp500_direction_prediction.ipynb   ← main notebook
│
├── outputs/
│   ├── figures/
│   │   ├── eda_overview.png
│   │   ├── model_comparison.png
│   │   ├── confusion_matrices.png
│   │   └── backtest.png
│   └── model_results.csv
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

## Models Compared

| # | Model | Scaling |
|---|---|---|
| 1 | Logistic Regression | StandardScaler (Pipeline) |
| 2 | Random Forest | None |
| 3 | HistGradientBoostingClassifier | None |
| 4 | XGBoost | None |
| 5 | LightGBM | None |
| 6 | K-Nearest Neighbors | StandardScaler (Pipeline) |
| 7 | Neural Network (MLPClassifier) | StandardScaler (Pipeline) |

Two naive baselines are included for comparison:
- **AlwaysUp** — predicts the market goes up every day
- **Momentum** — predicts the same direction as the previous day's return

---

## Features

All features are derived from historical price and volume data only (no external data sources):

- **Momentum returns**: 1-day, 5-day, 7-day, 21-day, 30-day percentage changes
- **Trend indicators**: Price-to-MA ratios (MA5, MA20, MA50, MA200)
- **Volatility**: Rolling standard deviation of daily returns (5, 20, 60-day windows)
- **Volume**: 1-day and 5-day volume percentage changes
- **Calendar**: Day of week, month

---

## Evaluation

- **Train/test split**: chronological — trained on 1990–2019, tested on 2020–present
- **Metrics**: Accuracy, Precision, Recall, F1-score, ROC-AUC
- **No random shuffling** — strictly time-based to avoid look-ahead bias

---

## Setup

```bash
pip install -r requirements.txt
jupyter notebook sp500_direction_prediction.ipynb
```

---

## Key Findings

Although some gradient boosting models achieved slightly higher ROC-AUC scores than the naive baselines, the overall improvement was limited. This is consistent with the efficient market hypothesis — short-term S&P 500 direction is difficult to predict from price history alone. More advanced features (VIX, treasury yields, macroeconomic indicators) may improve future versions.
