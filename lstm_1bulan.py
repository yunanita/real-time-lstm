# ==================== 0. IMPORT LIBRARIES ====================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from datetime import datetime
import pytz

from google.oauth2 import service_account
from google.cloud import bigquery

from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_ljungbox

from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error


# ==================== 1. KONFIGURASI ====================
import os
# Ambil path credential dari environment variable (untuk konsistensi dengan GitHub Actions)
SERVICE_ACCOUNT_PATH = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "time-series-analysis-480002-e7649b18ed82.json")
PROJECT_ID = "time-series-analysis-480002"
DATASET_ID = "SOL"
PREDICTION_DATASET = "PREDIKSI"

TIMEFRAME_CONFIG = {
    'SOL_1menit': {
        'table_name': 'SOL_1menit',
        'lookback_years': 1,
        'forecast_steps': 15,
        'timeframe_minutes': 1,
        'horizon_real': '15 menit',
        'exog_columns': ['volume'],
        'max_diff': 2,
        'p_range': range(0,3),
        'q_range': range(0,3)
    },
    'SOL_15menit': {
        'table_name': 'SOL_15menit',
        'lookback_years': 2,
        'forecast_steps': 8,
        'timeframe_minutes': 15,
        'horizon_real': '120 menit',
        'exog_columns': ['volume'],
        'max_diff': 2,
        'p_range': range(0,4),
        'q_range': range(0,4)
    },
    'SOL_1jam': {
        'table_name': 'SOL_1jam',
        'lookback_years': 3,
        'forecast_steps': 12,
        'timeframe_minutes': 60,
        'horizon_real': '12 jam',
        'exog_columns': ['volume'],
        'max_diff': 2,
        'p_range': range(0,4),
        'q_range': range(0,4)
    },
    'SOL_1hari': {
        'table_name': 'SOL_1hari',
        'lookback_years': 4,
        'forecast_steps': 1,
        'timeframe_minutes': 1440,
        'horizon_real': '1 hari',
        'exog_columns': ['volume'],
        'max_diff': 2,
        'p_range': range(0,3),
        'q_range': range(0,3)
    },
    'SOL_1bulan': {
        'table_name': 'SOL_1bulan',
        'lookback_years': None,
        'forecast_steps': 1,
        'timeframe_minutes': 43200,
        'horizon_real': '1 bulan',
        'exog_columns': ['volume'],
        'max_diff': 1,
        'p_range': range(0,3),
        'q_range': range(0,3)
    }
}


# ==================== 2. INISIALISASI BIGQUERY ====================
creds = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_PATH)
client = bigquery.Client(credentials=creds, project=creds.project_id)


# ==================== 3. FUNGSI PENDUKUNG ARIMAX ====================
def adf_test(series):
    return adfuller(series.dropna())[1]

def make_stationary(series, max_diff):
    d = 0
    s = series.copy()
    while d <= max_diff:
        pval = adf_test(s)
        if pval < 0.05:
            break
        s = s.diff().dropna()
        d += 1
    return d

def select_best_arimax(y, exog, d, p_range, q_range):
    best_aic = np.inf
    best_model = None
    best_order = None
    for p in p_range:
        for q in q_range:
            try:
                model = SARIMAX(
                    y, exog=exog,
                    order=(p,d,q),
                    enforce_stationarity=False,
                    enforce_invertibility=False
                )
                result = model.fit(disp=False)
                if result.aic < best_aic:
                    best_aic = result.aic
                    best_model = result
                    best_order = (p,d,q)
            except:
                continue
    return best_model, best_order, best_aic


# ==================== 4. PROSES PER TIMEFRAME ====================
def process_timeframe(tf_name, cfg):

    print(f"\n{'='*60}")
    print(f"PROCESSING {tf_name} (ARIMAX)")
    print(f"{'='*60}")

    table_ref = f"{PROJECT_ID}.{DATASET_ID}.{cfg['table_name']}"

    query = f"""
        SELECT timestamp, close, volume, datetime
        FROM `{table_ref}`
        ORDER BY timestamp
    """
    df = client.query(query).to_dataframe()

    if 'datetime' not in df.columns:
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')

    df = df.sort_values('datetime').drop_duplicates('datetime')

    y = df['close']
    exog = df[cfg['exog_columns']]

    # 1. Korelasi
    print("\n[1] Korelasi X-Y")
    print(df[['close'] + cfg['exog_columns']].corr())

    # 2–3. Stasioner & differencing
    d = make_stationary(y, cfg['max_diff'])
    print(f"\n[2–3] Differencing optimal d = {d}")

    # 4. ACF & PACF
    diff_series = y.diff(d).dropna()
    plot_acf(diff_series, lags=20)
    plot_pacf(diff_series, lags=20)
    plt.show()

    # 5–9. Model terbaik
    split = int(len(df)*0.8)
    model, order, aic = select_best_arimax(
        y[:split],
        exog[:split],
        d,
        cfg['p_range'],
        cfg['q_range']
    )

    print(f"\n[5–6–9] Model terbaik: ARIMAX{order} | AIC={aic:.2f}")

    # 7. Signifikansi
    print("\n[7] Uji Signifikansi Parameter")
    print(model.summary())

    # 8. Diagnostik residual
    lb = acorr_ljungbox(model.resid, lags=[10], return_df=True)
    print("\n[8] Uji Ljung-Box")
    print(lb)

    # 10. Forecasting
    last_exog = exog.iloc[-1].values
    exog_future = np.tile(last_exog, (cfg['forecast_steps'], 1))
    forecast = model.forecast(steps=cfg['forecast_steps'], exog=exog_future)

    # 11. Uji kelayakan
    print("\n[11] Uji Kelayakan Model")
    if lb['lb_pvalue'].iloc[0] > 0.05:
        print("✔ Model layak (residual white noise)")
    else:
        print("✗ Model belum layak")

    # Visualisasi
    plt.figure(figsize=(12,5))
    plt.plot(df['datetime'].iloc[-200:], y.iloc[-200:], label='Actual')
    future_dates = pd.date_range(
        start=df['datetime'].iloc[-1],
        periods=cfg['forecast_steps']+1,
        freq=f"{cfg['timeframe_minutes']}min"
    )[1:]
    plt.plot(future_dates, forecast, marker='o', label='Forecast')
    plt.title(f"{tf_name} - ARIMAX Forecast")
    plt.legend()
    plt.grid()
    plt.show()


# ==================== 5. MAIN ====================
def main():
    print("AUTOMATED SOL PRICE FORECASTING SYSTEM - ARIMAX")
    for tf_name, cfg in TIMEFRAME_CONFIG.items():
        process_timeframe(tf_name, cfg)


# ==================== 6. EKSEKUSI ====================
if __name__ == "__main__":
    print("MEMULAI PROSES FORECASTING ARIMAX...")
    main()
    print("SELESAI")