# S&P 500 Direction Prediction — ML Model Comparison

A machine learning portfolio project that compares seven classification models for predicting the **daily direction** of the S&P 500 index across three short-term horizons: **1 day**, **7 days**, and **30 days**.

> **Disclaimer:** This is an educational and portfolio project. Results should not be used for real trading decisions.

---

## Research Question

Can historical S&P 500 price patterns, technical indicators, and macroeconomic features predict whether the index will go up or down over the next 1, 7, or 30 trading days?

---

## Project Structure

```
trading_model/
│
├── sp500_direction_prediction.ipynb   ← main notebook
│
├── outputs/
│   ├── figures/
│   │   ├── eda_overview.png
│   │   ├── model_comparison.png
│   │   ├── confusion_matrices.png
│   │   ├── feature_importance.png
│   │   ├── backtest.png
│   │   └── backtest_dca.png
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

27 features derived from two sources:

**Price & Volume (16 features)**
- **Momentum returns**: 1-day, 5-day, 7-day, 21-day, 30-day percentage changes
- **Trend indicators**: Price-to-MA ratios (MA5, MA20, MA50, MA200)
- **Volatility**: Rolling standard deviation of daily returns (5, 20, 60-day windows)
- **Volume**: 1-day and 5-day volume percentage changes
- **Calendar**: Day of week, month

**External Macro (11 features)**
| Series | Ticker | Features |
|---|---|---|
| VIX | `^VIX` | Level, 1-day change, ratio to 20-day MA |
| 10Y Treasury Yield | `^TNX` | Level, yield curve spread (10Y − T-bill), 5-day change |
| 13-week T-bill | `^IRX` | Used to compute yield curve slope |
| US Dollar Index | `DX-Y.NYB` | 1-day return, ratio to 20-day MA |
| WTI Crude Oil | `CL=F` | 1-day return, 5-day return, ratio to 20-day MA |

---

## Evaluation

- **Train/test split**: chronological — trained on 1990–2019, tested on 2020–present
- **Hyperparameter tuning**: `RandomizedSearchCV` with `TimeSeriesSplit` (5 folds)
- **Metrics**: Accuracy, Precision, Recall, F1-score, ROC-AUC
- **No random shuffling** — strictly time-based to avoid look-ahead bias

---

## Backtesting

Two backtest strategies are compared against buy-and-hold using a $100 budget:

- **Model Strategy** — invest $100 on day one; hold when the best tuned model predicts up, move to cash when it predicts down
- **DCA (Dollar-Cost Averaging)** — invest equal monthly instalments spread over the full test period

The comparison illustrates the trade-off between lump-sum investing, active model-guided timing, and systematic DCA.

---

## Setup

```bash
pip install -r requirements.txt
jupyter notebook sp500_direction_prediction.ipynb
```

---

## Key Findings

Adding external macro features (VIX, Treasury yields, yield curve, DXY, oil) improved model performance over price-only features, with VIX and yield-related signals ranking highest in tree-based feature importance. However, overall ROC-AUC remained modest (0.52–0.58 on the 1-day horizon), consistent with the efficient market hypothesis — short-term S&P 500 direction is difficult to predict from publicly available data alone.

In the backtest, Buy & Hold outperformed both the model strategy and DCA over the 2020–present test window, highlighting that a model with limited predictive accuracy cannot reliably time the market well enough to beat a simple long position. DCA finished below Buy & Hold as expected in a rising market, but offers reduced timing risk for real-world investors.
