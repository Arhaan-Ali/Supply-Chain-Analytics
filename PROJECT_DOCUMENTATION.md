# Project Documentation — Predictive Analytics for Supply Chain

## Architecture Diagram

> Add your architecture image here.
>
> **Image placeholder:** `docs/architecture.png`

---

## Dataset Description

The project uses the following fields (Rossmann-like retail dataset):

- **Stores**
- **Sales**
- **Product Category** *(NA in this dataset / not available)*
- **Promo, Promo2, Promo2Since[Year/Week], PromoInterval**
- **Open** *(binary store open flag)*
- **StateHoliday, SchoolHoliday**
- **CompetitionDistance**

---

## Missing Values

### Test.csv

- **Issue**: 11 missing `Open` values across 11 unique `Id` rows.
- **Fix**: filled with `1`.
- **Reasoning**: `Open` is a binary operational flag; statistical imputation isn’t appropriate. Assumed stores are operational unless explicitly closed.

> Add your missing-values screenshot here.
>
> **Image placeholder:** `docs/missing_test_open.png`

### Train.csv

- **Issue**: none

### Store.csv

- **Issue**: 2343 missing values across 750 unique `Store` values in:
  - `Promo2`
  - competition open since month/year fields
  - `CompetitionDistance`
- **Fix**:
  - promotional/competition “presence” fields filled with `0` (interpreted as absence)
  - missing `CompetitionDistance` imputed with the **median** (stability + outlier-robust)

> Add your missing-values screenshot here.
>
> **Image placeholder:** `docs/missing_store.png`

---

## Merger

The training and testing datasets were enriched by integrating them with the store dataset to incorporate additional contextual information required for analysis.

- **Inputs**: `Data/train.csv`, `Data/test.csv`, `Data/store.csv`
- **Outputs**:
  - `Data/Processed Data/train_merged.csv`
  - `Data/Processed Data/test_merged.csv`

These consolidated datasets improve feature availability and consistency for subsequent processing and modeling tasks.

---

## Data Engineering

Data engineering techniques were applied to the consolidated dataset (`train_merged`) to enhance its analytical value. The resultant dataset is `train_features.csv`. This included the creation of derived features aimed at capturing temporal patterns, promotional effects, and sales dynamics.

### Engineered Features (high level)

- **Rolling averages**: `rolling_7`, `rolling_30`
- **Lag features**: `lag_1`, `lag_7`, `lag_14`
- **Promotion recency**: `days_since_promo`
- *(Additional features may exist depending on merged columns, e.g. holiday flags.)*

### Missing values after feature engineering

Initial missingness (post-feature engineering):

- **5,81,158 missing values** across **1,115** unique `Store` values

Imputation strategies:

- **Forward fill / backward fill** for sequential lag/rolling features to preserve temporal continuity.
- **Seasonal imputation** (where applicable) to account for recurring patterns and periodic trends.

Updated missing value status (after imputation pass):

- `rolling_7`: 6,690 (0.66%)
- `rolling_30`: 32,335 (3.18%)
- `lag_1`: 1,115 (0.11%)
- `lag_7`: 7,805 (0.77%)
- `lag_14`: 15,610 (1.53%)
- `days_since_promo`: 6,689 (0.66%)

Remaining total:

- **70,244 missing values** across **1,115** unique `Store` values

> Note: some “remaining” NaNs can be expected at the beginning of each store’s time series due to lag/rolling window definitions (e.g., the first 29 days cannot have a full `rolling_30`).

---

## EDA

### Graph 1 — Sales trend over time (5 stores)

> **Image placeholder:** `docs/eda_sales_trend_5stores.png`

**Insights**
- Weekly sales pattern with sharp drops likely due to closures (often Sundays).
- Store 726 performs best among shown stores (peaks at 30k+).
- Promotions likely drive many sales peaks.
- Wave-like behavior suggests strong seasonality; Prophet is expected to model this well.

### Graph 2 — Sales trend by store type

> **Image placeholder:** `docs/eda_sales_by_storetype.png`

**Insights**
- Store type **b** dominates sales.
- Types **a/c/d** show similar patterns (similar demand segments).
- Seasonality is consistent across store types.
- Store type **b** increases sales over time (growing demand / improved performance).

### Graph 3 — Seasonality decomposition (trend/seasonal/residual)

> **Image placeholder:** `docs/eda_seasonality_store_examples.png`

**Insights**
- Sales are volatile and influenced by closures, promotions, and external factors.
- Trend is relatively smooth (baseline demand stable).
- Strong weekly seasonality across stores, with periodic dips.

### Graph 4 — Promo vs no promo

> **Image placeholder:** `docs/eda_promo_impact.png`

**Insights**
- Promos show ~80% increase in sales (large positive impact).
- Promotions increase average sales and probability of high sales (also increase variability).
- No promo periods are more stable but lower revenue.

### Graph 5 — Correlation heatmap

> **Image placeholder:** `docs/eda_corr_heatmap.png`

**Insights**
- `Sales` vs `Customers` correlation ~0.89 (strong direct link).
- `Sales` correlated with lag features (`lag_7`, `lag_14`) ~0.67 (past sales predictive).
- `Sales` correlated with rolling averages ~0.57 (trend capture).
- `Promo` correlation ~0.45 (promotions increase sales).
- `days_since_promo` negative correlation ~-0.36 (sales decay after promo).
- Day-of-week correlation indicates low-performing days (often Sundays).
- Competition-related features show low impact.

### Graph 6 — Actual vs rolling averages

> **Image placeholder:** `docs/eda_actual_vs_rolling.png`

**Insights**
- Actual sales are noisy.
- `rolling_7` tracks short-term patterns well (useful for near-term forecasting).
- `rolling_30` captures broader trend (useful for longer horizon).

### Graph 7 — Day of week vs sales

> **Image placeholder:** `docs/eda_dow_vs_sales.png`

**Insights**
- Confirms weekly sales cycle.
- Highest earning days: Monday and Friday.
- Lowest earning day: Sunday (many zeros → stores often closed).

---

## Demand Forecasting — Facebook Prophet

Initially, a global forecasting approach was implemented by training a single Prophet model across all stores. This resulted in poor performance (MAPE ~55%) due to high variability across stores (scale, promo effects, local demand behavior).

To address this, a **store-wise** strategy was adopted: one Prophet model per store. This allowed each model to learn store-specific trend and seasonality, leading to major improvement.

### Performance summary (store-level Prophet)

- **MAPE** — min: 2.86% | mean: 6.11% | max: 32.06%  
- **MAE** — min: 134.43 | mean: 360.93 | max: 2533.25  
- **R²** — min: -0.2101 | mean: 0.9640 | max: 0.9936

Top 5 best stores (lowest MAPE):

| Store | MAE | RMSE | MAPE | R² |
|---:|---:|---:|---:|---:|
| 1092 | 312.76 | 464.96 | 2.86% | 0.9917 |
| 539  | 295.09 | 408.96 | 2.98% | 0.9923 |
| 1033 | 369.70 | 504.63 | 3.02% | 0.9932 |
| 532  | 347.16 | 463.50 | 3.25% | 0.9903 |
| 842  | 544.80 | 795.11 | 3.26% | 0.9910 |

Top 5 worst stores (highest MAPE):

| Store | MAE | RMSE | MAPE | R² |
|---:|---:|---:|---:|---:|
| 931  | 1147.03 | 1386.97 | 32.06% | 0.1416 |
| 1099 | 2286.68 | 2758.10 | 31.07% | -0.2101 |
| 1045 | 1987.28 | 2403.49 | 26.92% | 0.3309 |
| 708  | 1170.99 | 1466.47 | 23.95% | 0.6398 |
| 364  | 994.43  | 1258.52 | 23.88% | 0.6661 |

> Add your Prophet results screenshot here.
>
> **Image placeholder:** `docs/prophet_results.png`

---

## Demand Forecasting — XGBoost (Gradient Boosted Trees)

### Why XGBoost

Prophet captures trend/seasonality well, but tree-based models can excel when:
- relationships are nonlinear,
- promotions and interactions matter,
- lag/rolling features provide strong predictive signal.

XGBoost is used here as a supervised regression model over engineered tabular features.

### Key modeling approach

- **Target transform**: `Sales_log = log1p(Sales)` to stabilize variance and reduce the impact of extreme spikes.
- **Chronological split**: data is split by time (first 80% train, last 20% test) to reflect forecasting conditions.
- **Core features** (as used in `Training/XGBoostTrain.py`):
  - `Customers`
  - `rolling_7`, `rolling_30`
  - `lag_1`, `lag_7`, `lag_14`
  - `days_since_promo`
  - `Promo`
  - `Store`
  - `DayOfWeek`

### Hyperparameter tuning

Optuna is used to tune parameters (20 trials), minimizing RMSE on the validation split. The best parameters are saved for reproducibility.

### Explainability (SHAP)

SHAP values are computed over a sample of the test set to:
- identify the most influential drivers of predictions,
- validate the model is leveraging meaningful signals (e.g., customers, promo, lag).

### Outputs

Training produces:

- **Model artifact**: `Models/xgboost_model.pkl`
- **Best params**: `Models/xgboost_best_params.csv`
- **Plots**:
  - `Data/Processed Data/Graphs/xgboost_actual_vs_predicted.png`
  - `Data/Processed Data/Graphs/shap_summary.png`
  - `Data/Processed Data/Graphs/shap_bar.png`

> Add your XGBoost plots here.
>
> **Image placeholders:**
> - `docs/xgb_actual_vs_pred.png`
> - `docs/shap_summary.png`
> - `docs/shap_bar.png`

---

## Production Forecast Generation (Prophet + XGBoost)

The script `Production/Forecasting.py` generates future forecasts using the trained models.

### Inputs

- `Models/prophet_models.pkl` (store-wise Prophet models)
- `Models/xgboost_model.pkl` (trained XGBoost regressor)
- `Data/Processed Data/train_features.csv` (latest historical feature data)

### Prophet forecasting (30/90 days)

For each store:
- build a 90-day daily future frame starting after the last training date,
- set baseline future regressors (e.g., `Open=1`, `Promo=0`, and derived rolling/lag approximations from recent history),
- predict with two interval widths:
  - 80% (lower_80, upper_80)
  - 95% (lower_95, upper_95)

Saved outputs:

- `Data/Results/prophet_all_stores_30d.csv`
- `Data/Results/prophet_all_stores_90d.csv`

### XGBoost forecasting (next day and next 7 days)

XGBoost uses a **recursive** strategy:
- predict day \(t+1\),
- append predicted sales to history,
- recompute lag/rolling features from updated history,
- repeat for the requested horizon.

Saved outputs:

- `Data/Results/xgb_all_stores_next_day.csv`
- `Data/Results/xgb_all_stores_7day.csv`

> Add any forecasting output screenshots here.
>
> **Image placeholder:** `docs/production_forecasts.png`

---

## Inventory Optimisation

The script `Production/Inventory_Optimisation.py` converts demand forecasts into inventory decisions.

### Goal

For each store:
- estimate demand variability,
- compute safety stock and reorder point (ROP),
- compute EOQ (economic order quantity),
- determine which stores need reordering now.

### Method

Demand statistics (from historical sales):
- average daily demand
- standard deviation of daily demand
- annualized demand (\(\text{avg\_daily} \times 365\))

Safety stock (95% service level):
\[
\text{safety\_stock} = Z_{0.95}\cdot \sigma_d \cdot \sqrt{L}
\]

Reorder point:
\[
\text{ROP} = (\mu_d \cdot L) + \text{safety\_stock}
\]

EOQ:
\[
\text{EOQ} = \sqrt{\frac{2DS}{H}}
\]

Where:
- \(Z_{0.95} = 1.645\)
- \(L\) = lead time (days)
- \(D\) = annual demand
- \(S\) = order cost
- \(H\) = holding cost per unit per year

### Current stock simulation (project assumption)

Because real on-hand inventory is not present in the dataset, current stock is simulated as a random multiple of ROP per store. This enables a full inventory pipeline demo but should be replaced with real inventory feeds in production.

### Outputs

- `Data/Inventory/weekly_replenishment_report.csv`
- `Data/Inventory/reorder_now.csv`
- Plots:
  - `Data/Inventory/stock_vs_rop.png`
  - `Data/Inventory/inventory_status_pie.png`
  - `Data/Inventory/eoq_distribution.png`

> Add your inventory plots here.
>
> **Image placeholders:**
> - `docs/inventory_stock_vs_rop.png`
> - `docs/inventory_status_pie.png`
> - `docs/inventory_eoq_distribution.png`

---

## Dashboard (Analytics & Decision View)

The script `Dashboard/Dashboard.py` generates a multi-panel dashboard summarizing:
- actual sales history,
- Prophet and XGBoost forecasts,
- store KPIs and forecast accuracy,
- inventory actions,
- SHAP-based feature importance.

### Inputs

- `Data/Processed Data/train_features.csv`
- Forecast files:
  - `Data/Results/prophet_all_stores_30d.csv`
  - `Data/Results/prophet_all_stores_90d.csv`
  - `Data/Results/xgb_all_stores_next_day.csv`
  - `Data/Results/xgb_all_stores_7day.csv`
- Inventory report:
  - `Data/Inventory/weekly_replenishment_report.csv`
- Model for SHAP:
  - `Models/xgboost_model.pkl`

### Panels generated

1. **Demand Forecast View** (actual vs Prophet vs XGBoost for a sample store)  
   Output: `Dashboard/panel1_demand_forecast.png`
2. **Store Performance KPIs** (top stores by revenue + MAPE distribution)  
   Output: `Dashboard/panel2_store_kpis.png`
3. **Inventory Status Table** (top stores; reorder-now highlighted)  
   Output: `Dashboard/panel3_inventory_table.png`
4. **Seasonality Heatmap** (avg sales by month × weekday)  
   Output: `Dashboard/panel4_seasonality_heatmap.png`
5. **SHAP Feature Importance** (global importance)  
   Output: `Dashboard/panel5_shap_importance.png`

Combined:
- `Dashboard/full_dashboard.png`

> Add your dashboard screenshots here.
>
> **Image placeholders:**
> - `docs/dashboard_panel1.png`
> - `docs/dashboard_panel2.png`
> - `docs/dashboard_panel3.png`
> - `docs/dashboard_panel4.png`
> - `docs/dashboard_panel5.png`
> - `docs/dashboard_full.png`

---

## How to Run (pipeline)

Install:

```bash
pip install -r requirements.txt
```

Run end-to-end (from repo root). Because scripts use `../Data/...` paths, run them from their folder:

```bash
cd research && python preprocessing.py && python merger.py && cd ..
cd Production && python Data_engineering.py && python Imputation.py && cd ..
cd Training && python ProphetTrain.py && python XGBoostTrain.py && cd ..
cd Production && python Forecasting.py && python Inventory_Optimisation.py && cd ..
cd Dashboard && python Dashboard.py && cd ..
```

