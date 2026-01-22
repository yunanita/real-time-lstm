# ==================== LSTM PREDICTION - 1 jam ====================
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
import json  
import os     

from google.oauth2 import service_account
from google.cloud import bigquery
import google.auth  

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow as tf

import warnings
warnings.filterwarnings('ignore')

# ==================== FUNGSI AUTHENTIKASI ====================
def get_credentials():
    """
    Mendapatkan credentials dari:
    1. Environment variable GCP_CREDS (untuk GitHub Actions)
    2. File JSON lokal (untuk development lokal)
    """
    gcp_creds_json = os.environ.get("GCP_CREDS")  # bisa ganti "GCP_CREDENTIALS" jika perlu
    
    if gcp_creds_json:
        try:
            creds_dict = json.loads(gcp_creds_json)
            credentials = service_account.Credentials.from_service_account_info(creds_dict)
            print("✅ Menggunakan credentials dari environment variable GCP_CREDS")
            return credentials
        except json.JSONDecodeError as e:
            print(f"❌ Error parsing JSON dari environment variable: {e}")
            raise
    
    # Fallback: Coba dari file lokal
    local_file = "time-series-analysis-480002-e7649b18ed82.json"
    if os.path.exists(local_file):
        credentials = service_account.Credentials.from_service_account_file(local_file)
        print(f"✅ Menggunakan credentials dari file lokal: {local_file}")
        return credentials
    
    # Fallback terakhir
    try:
        credentials, _ = google.auth.default()
        print("⚠️  Menggunakan default application credentials")
        return credentials
    except Exception as e:
        print(f"❌ Error menggunakan default credentials: {e}")
    
    raise Exception("❌ Tidak ditemukan credentials! Set environment variable GCP_CREDS atau sediakan file JSON.")

# ==================== KONFIGURASI ====================
PROJECT_ID = "time-series-analysis-480002"
DATASET_ID = "SOL"
PREDICTION_DATASET = "PREDIKSI"

# Konfigurasi 1 jam - OPTIMIZED
TIMEFRAME_CONFIG = {
    '1jam': {
        'table_name': 'SOL_1jam',
        'lookback_years': 3,
        'retrain_frequency': '2x/hari',
        'retrain_times': ['08:00', '20:00'],
        'forecast_steps': 12,
        'timeframe_minutes': 60,
        'horizon_real': '12 jam',
        'sequence_length': 24,      # ⬇️ OPTIMIZED
        'lstm_units': 64,           # ⬇️ OPTIMIZED
        'dropout': 0.2,
        'max_epochs': 30,           # ⬇️ OPTIMIZED
        'patience': 8,              # ⬇️ OPTIMIZED
        'batch_size': 32           
    }
}

# ==================== INISIALISASI BIGQUERY ====================
try:
    credentials = get_credentials()
    client = bigquery.Client(credentials=credentials, project=PROJECT_ID)
    print(f"✅ Berhasil terkoneksi ke BigQuery. Project: {PROJECT_ID}")
except Exception as e:
    print(f"❌ Gagal menginisialisasi BigQuery client: {e}")
    client = None  # Untuk testing tanpa koneksi

# ==================== FUNGSI LSTM ====================
def create_sequences(data, sequence_length):
    """Membuat sequences untuk LSTM"""
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i+sequence_length])
        y.append(data[i+sequence_length])
    return np.array(X), np.array(y)

def build_lstm_model(sequence_length, units=64, dropout=0.2):
    """Membangun model LSTM yang optimal"""
    model = Sequential([
        Input(shape=(sequence_length, 1)),
        LSTM(units, return_sequences=True),
        Dropout(dropout),
        LSTM(units // 2),
        Dropout(dropout),
        Dense(1)
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    
    return model

def forecast_lstm(model, last_sequence, forecast_steps, scaler):
    """Melakukan forecasting dengan LSTM"""
    predictions = []
    current_sequence = last_sequence.copy()
    
    for _ in range(forecast_steps):
        X_input = current_sequence.reshape(1, -1, 1)
        pred_scaled = model.predict(X_input, verbose=0)[0, 0]
        predictions.append(pred_scaled)
        current_sequence = np.append(current_sequence[1:], pred_scaled)
    
    predictions = np.array(predictions).reshape(-1, 1)
    predictions_denorm = scaler.inverse_transform(predictions)
    
    return predictions_denorm.flatten()

# ==================== FUNGSI BIGQUERY ====================
def delete_and_recreate_table(table_name, schema):
    """Hapus tabel dan buat ulang dengan schema yang sama"""
    table_id = f"{PROJECT_ID}.{PREDICTION_DATASET}.{table_name}"
    
    try:
        client.delete_table(table_id)
        print(f"  ✓ Tabel {table_name} lama dihapus")
    except:
        print(f"  ℹ Tabel {table_name} belum ada")
    
    table = bigquery.Table(table_id, schema=schema)
    table = client.create_table(table)
    print(f"  ✓ Tabel {table_name} baru dibuat")
    
    return table_id

def save_all_data_complete(timeframe_name, df_actual, forecasts, metrics, train_test_split_idx):
    """Simpan SEMUA DATA tanpa sampling dan tanpa limit"""
    current_time = datetime.now(pytz.timezone('Asia/Jakarta'))
    current_time_utc = current_time.astimezone(pytz.UTC)
    
    records = []
    total_records = len(df_actual) + len(forecasts)
    
    print(f"  Menyimpan {total_records} records ke BigQuery...")
    print(f"    - Training data: {train_test_split_idx} records")
    print(f"    - Testing data: {len(df_actual) - train_test_split_idx} records")
    print(f"    - Forecast data: {len(forecasts)} records")
    
    # ========== 1. TRAINING DATA (SEMUA) ==========
    print("    1. Menyimpan training data...")
    for i in range(train_test_split_idx):
        if i % 1000 == 0:  # Progress indicator
            print(f"       Progress: {i}/{train_test_split_idx} training records")
        
        timestamp = df_actual['datetime'].iloc[i]
        if pd.isna(timestamp):
            timestamp_utc = current_time_utc
        elif timestamp.tz is None:
            timestamp_utc = timestamp.tz_localize('UTC')
        else:
            timestamp_utc = timestamp.astimezone(pytz.UTC)
        
        records.append({
            'timestamp': timestamp_utc,
            'price': float(df_actual['close'].iloc[i]),
            'data_type': 'TRAIN',
            'model_type': 'LSTM',
            'timeframe': timeframe_name,
            'training_date': current_time_utc.date(),
            'created_at': current_time_utc,
            'mape': float(metrics['mape']),
            'accuracy': float(metrics['accuracy']),
            'mse': float(metrics['mse']),
            'rmse': float(metrics['rmse']),
            'mae': float(metrics['mae'])
        })
    
    # ========== 2. TESTING DATA (SEMUA) ==========
    print("    2. Menyimpan testing data...")
    for i in range(train_test_split_idx, len(df_actual)):
        if i % 100 == 0:  # Progress indicator
            print(f"       Progress: {i-train_test_split_idx}/{len(df_actual)-train_test_split_idx} testing records")
        
        timestamp = df_actual['datetime'].iloc[i]
        if pd.isna(timestamp):
            timestamp_utc = current_time_utc
        elif timestamp.tz is None:
            timestamp_utc = timestamp.tz_localize('UTC')
        else:
            timestamp_utc = timestamp.astimezone(pytz.UTC)
        
        records.append({
            'timestamp': timestamp_utc,
            'price': float(df_actual['close'].iloc[i]),
            'data_type': 'TEST',
            'model_type': 'LSTM',
            'timeframe': timeframe_name,
            'training_date': current_time_utc.date(),
            'created_at': current_time_utc,
            'mape': float(metrics['mape']),
            'accuracy': float(metrics['accuracy']),
            'mse': float(metrics['mse']),
            'rmse': float(metrics['rmse']),
            'mae': float(metrics['mae'])
        })
    
    # ========== 3. FORECAST DATA ==========
    print("    3. Menyimpan forecast data...")
    config = TIMEFRAME_CONFIG[timeframe_name]
    
    for i, forecast_price in enumerate(forecasts, 1):
        # Generate forecast date berdasarkan timeframe
        forecast_date = current_time + timedelta(hours=i)  # 1 jam ke depan
        forecast_date_utc = forecast_date.astimezone(pytz.UTC)
        
        records.append({
            'timestamp': forecast_date_utc,
            'price': float(forecast_price),
            'data_type': 'FORECAST',
            'model_type': 'LSTM',
            'timeframe': timeframe_name,
            'training_date': current_time_utc.date(),
            'created_at': current_time_utc,
            'mape': float(metrics['mape']),
            'accuracy': float(metrics['accuracy']),
            'mse': float(metrics['mse']),
            'rmse': float(metrics['rmse']),
            'mae': float(metrics['mae'])
        })
    
    # ========== 4. CONVERT TO DATAFRAME ==========
    df_all = pd.DataFrame(records)
    print(f"    ✓ Data prepared: {len(df_all)} records")
    
    # ========== 5. DEFINE SCHEMA ==========
    schema = [
        bigquery.SchemaField("timestamp", "TIMESTAMP", mode="REQUIRED"),
        bigquery.SchemaField("price", "FLOAT64", mode="REQUIRED"),
        bigquery.SchemaField("data_type", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("model_type", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("timeframe", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("training_date", "DATE", mode="REQUIRED"),
        bigquery.SchemaField("created_at", "TIMESTAMP", mode="REQUIRED"),
        bigquery.SchemaField("mape", "FLOAT64", mode="NULLABLE"),
        bigquery.SchemaField("accuracy", "FLOAT64", mode="NULLABLE"),
        bigquery.SchemaField("mse", "FLOAT64", mode="NULLABLE"),
        bigquery.SchemaField("rmse", "FLOAT64", mode="NULLABLE"),
        bigquery.SchemaField("mae", "FLOAT64", mode="NULLABLE")
    ]
    
    # ========== 6. SAVE TO BIGQUERY ==========
    table_name = f"lstm_{timeframe_name}"
    table_id = delete_and_recreate_table(table_name, schema)
    
    # Config untuk upload besar
    job_config = bigquery.LoadJobConfig(
        schema=schema,
        write_disposition=bigquery.WriteDisposition.WRITE_APPEND,
        max_bad_records=10  # Allow some bad records
    )
    
    try:
        print(f"    Mengupload {len(df_all)} records ke BigQuery...")
        job = client.load_table_from_dataframe(df_all, table_id, job_config=job_config)
        job.result()
        
        # Verifikasi data masuk
        count_query = f"SELECT COUNT(*) as cnt FROM `{table_id}`"
        count_result = client.query(count_query).to_dataframe()
        
        print(f"  ✓ Data berhasil disimpan ke {table_name}")
        print(f"     Total rows: {count_result['cnt'].iloc[0]:,}")
        print(f"     Training data: {train_test_split_idx:,} rows")
        print(f"     Testing data: {len(df_actual) - train_test_split_idx:,} rows")
        print(f"     Forecast data: {len(forecasts):,} rows")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Error menyimpan data: {str(e)}")
        return False

# ==================== PROSES UTAMA ====================
def process_1jam_full_data(force_retrain=True):
    """Proses prediksi untuk timeframe 1 jam dengan SEMUA DATA"""
    print("="*70)
    print("LSTM PREDICTION - SOL 1 JAM (FULL DATA VERSION)")
    print("="*70)
    
    timeframe_name = '1jam'
    config = TIMEFRAME_CONFIG[timeframe_name]
    
    # 1. LOAD DATA DARI BIGQUERY - TANPA LIMIT
    print(f"\n1. Memuat data dari {config['table_name']} (TANPA LIMIT)...")
    
    table_ref = f"{PROJECT_ID}.{DATASET_ID}.{config['table_name']}"
    
    try:
        # Hitung tanggal 3 tahun terakhir
        lookback_date = (datetime.now() - timedelta(days=config['lookback_years']*365)).strftime('%Y-%m-%d')
        
        # QUERY TANPA LIMIT - AMBIL SEMUA DATA
        query = f"""
            SELECT datetime, close
            FROM `{table_ref}`
            WHERE datetime >= '{lookback_date}'
            ORDER BY datetime
        """
        
        df = client.query(query).to_dataframe()
        print(f"   ✓ Data loaded: {len(df):,} rows (sejak {lookback_date})")
        
        if len(df) == 0:
            print("   ✗ Data kosong")
            return None
            
    except Exception as e:
        print(f"   ✗ Error loading data: {str(e)}")
        return None
    
    # 2. PREPROCESSING
    print(f"\n2. Preprocessing data...")
    
    df['datetime'] = pd.to_datetime(df['datetime'])
    if df['datetime'].dt.tz is None:
        df['datetime'] = df['datetime'].dt.tz_localize('UTC')
    
    df = df.sort_values('datetime').reset_index(drop=True)
    df = df.drop_duplicates(subset=['datetime'])
    
    print(f"   Data range: {df['datetime'].min()} to {df['datetime'].max()}")
    print(f"   Data points: {len(df):,}")
    print(f"   Last close price: {df['close'].iloc[-1]:.4f}")
    
    if len(df) < config['sequence_length'] * 2:
        print(f"   ✗ Data tidak cukup")
        return None
    
    # 3. NORMALISASI
    print(f"\n3. Normalisasi data...")
    close_prices = df['close'].values.reshape(-1, 1)
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    close_scaled = scaler.fit_transform(close_prices)
    print(f"   ✓ Data dinormalisasi")
    
    # 4. PREPARE SEQUENCES (SEMUA SEQUENCES)
    print(f"\n4. Membuat sequences (SEMUA)...")
    X, y = create_sequences(close_scaled, config['sequence_length'])
    
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    train_test_split_idx = train_size + config['sequence_length']
    
    print(f"   ✓ Sequences created")
    print(f"     Total sequences: {len(X):,}")
    print(f"     Train sequences: {len(X_train):,}")
    print(f"     Test sequences: {len(X_test):,}")
    
    # 5. TRAIN MODEL (OPTIMIZED)
    print(f"\n5. Training model LSTM (optimized)...")
    model_path = f"/tmp/lstm_model_{timeframe_name}.h5"
    
    if force_retrain or not os.path.exists(model_path):
        print(f"   Training model baru dengan parameter optimal...")
        
        model = build_lstm_model(config['sequence_length'], units=config['lstm_units'])
        
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=config['patience'],
            restore_best_weights=True,
            verbose=0
        )
        
        model_checkpoint = ModelCheckpoint(
            model_path,
            monitor='val_loss',
            save_best_only=True,
            verbose=0
        )
        
        print(f"   Training dengan {config['max_epochs']} epochs, batch size: {config['batch_size']}")
        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=config['max_epochs'],
            batch_size=config['batch_size'],
            verbose=0,
            callbacks=[early_stopping, model_checkpoint]
        )
        
        final_epoch = len(history.history['loss'])
        print(f"   ✓ Model trained ({final_epoch} epochs)")
        
    else:
        print(f"   Memuat model yang sudah ada...")
        model = load_model(model_path)
        print(f"   ✓ Model loaded from {model_path}")
    
    # 6. EVALUASI MODEL
    print(f"\n6. Evaluasi model...")
    
    y_pred_scaled = model.predict(X_test, verbose=0)
    y_pred = scaler.inverse_transform(y_pred_scaled)
    y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))
    
    mse = mean_squared_error(y_test_actual, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test_actual, y_pred)
    mape = mean_absolute_percentage_error(y_test_actual, y_pred) * 100
    accuracy = max(0, 100 - mape)
    
    print(f"   MSE:  {mse:.4f}")
    print(f"   RMSE: {rmse:.4f}")
    print(f"   MAE:  {mae:.4f}")
    print(f"   MAPE: {mape:.2f}%")
    print(f"   Akurasi: {accuracy:.2f}%")
    
    # 7. FORECASTING
    print(f"\n7. Forecasting {config['forecast_steps']} steps...")
    
    last_sequence = close_scaled[-config['sequence_length']:]
    forecasts = forecast_lstm(model, last_sequence, config['forecast_steps'], scaler)
    
    print(f"   ✓ Forecast generated:")
    for i, forecast in enumerate(forecasts, 1):
        hour = i
        print(f"     Step {i} ({hour} jam): {forecast:.4f}")
    
    last_actual = df['close'].iloc[-1]
    first_forecast = forecasts[0]
    if last_actual > 0:
        pct_change = ((first_forecast - last_actual) / last_actual) * 100
        print(f"   Perubahan: {last_actual:.4f} → {first_forecast:.4f} ({pct_change:+.2f}%)")
    
    # 8. SAVE ALL DATA TO BIGQUERY
    print(f"\n8. Menyimpan SEMUA DATA ke BigQuery...")
    
    metrics_dict = {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'mape': mape,
        'accuracy': accuracy
    }
    
    success = save_all_data_complete(
        timeframe_name=timeframe_name,
        df_actual=df,
        forecasts=forecasts,
        metrics=metrics_dict,
        train_test_split_idx=train_test_split_idx
    )
    
    if not success:
        return None
    
    # 9. FINAL SUMMARY
    print(f"\n" + "="*70)
    print("SUMMARY - 1 JAM (FULL DATA)")
    print("="*70)
    
    summary_data = {
        'Last Actual Price': f"{last_actual:.4f}",
        'Next Hour Forecast': f"{first_forecast:.4f}",
        '12th Hour Forecast': f"{forecasts[-1]:.4f}",
        'Change % (1h)': f"{pct_change:+.2f}%",
        'MAPE': f"{mape:.2f}%",
        'Accuracy': f"{accuracy:.2f}%",
        'Training Sequences': f"{len(X_train):,}",
        'Testing Sequences': f"{len(X_test):,}",
        'Total Data Points': f"{len(df):,}",
        'Forecast Steps': f"{config['forecast_steps']} (12 jam)",
        'Model Status': "Retrained" if (force_retrain or not os.path.exists(model_path)) else "Loaded",
        'Data Saved': f"COMPLETE ({len(df) + len(forecasts):,} rows)"
    }
    
    summary_df = pd.DataFrame(list(summary_data.items()), columns=['Metric', 'Value'])
    print(summary_df.to_string(index=False))
    
    return {
        'timeframe': timeframe_name,
        'forecasts': forecasts,
        'metrics': metrics_dict,
        'last_price': df['close'].iloc[-1],
        'first_forecast': forecasts[0] if len(forecasts) > 0 else None,
        'total_data_points': len(df)
    }

# ==================== JALANKAN ====================
if __name__ == "__main__":
    print("\nMEMULAI PREDIKSI LSTM UNTUK 1 JAM (FULL DATA)...")
    
    # Pastikan dataset PREDIKSI ada
    dataset_id = f"{PROJECT_ID}.{PREDICTION_DATASET}"
    try:
        client.get_dataset(dataset_id)
        print(f"✓ Dataset {PREDICTION_DATASET} sudah ada")
    except:
        print(f"Membuat dataset {PREDICTION_DATASET}...")
        dataset = bigquery.Dataset(dataset_id)
        dataset.location = "US"
        client.create_dataset(dataset)
        print(f"✓ Dataset {PREDICTION_DATASET} dibuat")
    
    # Jalankan proses
    result = process_1jam_full_data(force_retrain=True)
    
    if result:
        print("\n" + "="*70)
        print("PREDIKSI 1 JAM SELESAI")
        print("="*70)
        print(f"✓ SEMUA DATA telah disimpan ke BigQuery")
        print(f"✓ Tabel: {PROJECT_ID}.{PREDICTION_DATASET}.lstm_1jam")
        print(f"✓ Total rows: {result['total_data_points'] + len(result['forecasts']):,}")
        print(f"✓ Forecast 1 jam: {result['first_forecast']:.4f}")
        print(f"✓ Forecast 12 jam: {result['forecasts'][-1]:.4f}")
        print(f"✓ Accuracy: {result['metrics']['accuracy']:.2f}%")
    else:
        print("\n" + "="*70)
        print("PREDIKSI 1 JAM GAGAL")
        print("="*70)