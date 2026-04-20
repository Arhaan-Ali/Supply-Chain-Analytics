# ============================================================
# decomposition.py  —  Plot forecast decomposition
# ============================================================

import pickle
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# ── Load Models ───────────────────────────────────────────────────────────────
print("Loading models...")
with open("../Models/prophet_models.pkl", "rb") as f:
    store_models = pickle.load(f)
print(f"✅ Loaded {len(store_models)} store models")

# ── Load Data ─────────────────────────────────────────────────────────────────
df = pd.read_csv("../Data/Processed Data/train_features.csv")
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values(['Store', 'Date']).reset_index(drop=True)

REGRESSORS = [
    'Open', 'DayOfWeek', 'Customers', 'Promo',
    'days_since_promo', 'rolling_7', 'lag_7', 'lag_14', 'rolling_30'
]

# ── Helper: Build future dataframe ────────────────────────────────────────────
def build_future_df(store_df, periods):
    last_date    = store_df['Date'].max()
    future_dates = pd.date_range(
        start   = last_date + pd.Timedelta(days=1),
        periods = periods,
        freq    = 'D'
    )

    future = pd.DataFrame({'ds': future_dates})
    future['Open']             = 1
    future['DayOfWeek']        = future['ds'].dt.dayofweek + 1
    future['Customers']        = store_df['Customers'].rolling(7).mean().iloc[-1]
    future['Promo']            = 0
    future['days_since_promo'] = range(1, periods + 1)
    future['rolling_7']        = store_df['Sales'].tail(7).mean()
    future['rolling_30']       = store_df['Sales'].tail(30).mean()
    future['lag_7']            = store_df['Sales'].tail(7).mean()
    future['lag_14']           = store_df['Sales'].tail(14).mean()

    return future

# ── Plot Decomposition ────────────────────────────────────────────────────────
sample_store = 1
model        = store_models[sample_store]
store_df     = df[df['Store'] == sample_store].sort_values('Date').reset_index(drop=True)

future               = build_future_df(store_df, 90)
model.interval_width = 0.95
forecast             = model.predict(future)

fig = model.plot_components(forecast)
plt.suptitle(f"Forecast Decomposition — Store {sample_store}", fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig("../Data/Forecasts/decomposition_store_1.png", dpi=150, bbox_inches='tight')
plt.show()

print("✅ Decomposition plot saved → ../Data/Forecasts/decomposition_store_1.png")