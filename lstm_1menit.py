# ==================== LSTM PREDICTION - 1 Menit ====================
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

# Konfigurasi 1 menit - FULL DATA
TIMEFRAME_CONFIG = {
    '1menit': {
        'table_name': 'SOL_1menit',
        'lookback_years': 1,          # 1 TAHUN PENUH
        'retrain_frequency': '2x/hari',
        'retrain_times': ['08:00', '20:00'],
        'forecast_steps': 15,         # FORECAST 15 MENIT
        'timeframe_minutes': 1,
        'horizon_real': '15 menit',
        'sequence_length': 60,        # 1 JAM HISTORY
        'lstm_units': 64,             # 64 UNITS
        'dropout': 0.2,
        'max_epochs': 15,             # 15 EPOCHS
        'patience': 5,
        'batch_size': 512             # BATCH BESAR UNTUK CEPAT
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
    """Membuat sequences untuk LSTM - SEMUA SEQUENCES"""
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i+sequence_length])
        y.append(data[i+sequence_length])
    return np.array(X), np.array(y)

def build_lstm_model(sequence_length, units=64, dropout=0.2):
    """Membangun model LSTM dengan 64 units"""
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

def save_data_all(timeframe_name, df_actual, forecasts, metrics, train_test_split_idx):
    """Simpan SEMUA DATA tanpa sampling"""
    current_time = datetime.now(pytz.timezone('Asia/Jakarta'))
    current_time_utc = current_time.astimezone(pytz.UTC)
    
    records = []
    
    print(f"  Menyimpan SEMUA DATA ke BigQuery...")
    
    # ========== 1. TRAINING DATA (SEMUA) ==========
    print(f"    1. Menyimpan training data ({train_test_split_idx:,} records)...")
    
    # Untuk menghindari memory error, kita bagi menjadi batch
    batch_size = 50000
    num_batches = (train_test_split_idx + batch_size - 1) // batch_size
    
    for batch_num in range(num_batches):
        start_idx = batch_num * batch_size
        end_idx = min((batch_num + 1) * batch_size, train_test_split_idx)
        
        print(f"      Batch {batch_num+1}/{num_batches}: {start_idx:,}-{end_idx:,}")
        
        for idx in range(start_idx, end_idx):
            timestamp = df_actual['datetime'].iloc[idx]
            if pd.isna(timestamp):
                timestamp_utc = current_time_utc
            elif timestamp.tz is None:
                timestamp_utc = timestamp.tz_localize('UTC')
            else:
                timestamp_utc = timestamp.astimezone(pytz.UTC)
            
            records.append({
                'timestamp': timestamp_utc,
                'price': float(df_actual['close'].iloc[idx]),
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
    print(f"    2. Menyimpan testing data ({len(df_actual) - train_test_split_idx:,} records)...")
    
    for idx in range(train_test_split_idx, len(df_actual)):
        timestamp = df_actual['datetime'].iloc[idx]
        if pd.isna(timestamp):
            timestamp_utc = current_time_utc
        elif timestamp.tz is None:
            timestamp_utc = timestamp.tz_localize('UTC')
        else:
            timestamp_utc = timestamp.astimezone(pytz.UTC)
        
        records.append({
            'timestamp': timestamp_utc,
            'price': float(df_actual['close'].iloc[idx]),
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
        forecast_date = current_time + timedelta(minutes=i)
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
    print(f"    ✓ Data prepared: {len(df_all):,} records")
    
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
    
    job_config = bigquery.LoadJobConfig(
        schema=schema,
        write_disposition=bigquery.WriteDisposition.WRITE_APPEND
    )
    
    try:
        print(f"    Mengupload {len(df_all):,} records ke BigQuery...")
        job = client.load_table_from_dataframe(df_all, table_id, job_config=job_config)
        job.result()
        
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
def process_1menit_all_data(force_retrain=True):
    """Proses prediksi untuk timeframe 1 menit dengan SEMUA DATA TANPA STRIDE"""
    print("="*80)
    print("LSTM PREDICTION - SOL 1 MENIT")
    print("FULL DATA - NO STRIDE - 15 EPOCHS - 64 UNITS")
    print("="*80)
    
    timeframe_name = '1menit'
    config = TIMEFRAME_CONFIG[timeframe_name]
    
    # 1. LOAD DATA 1 TAHUN PENUH - TANPA LIMIT
    print(f"\n1. Memuat data 1 TAHUN PENUH dari {config['table_name']}...")
    
    table_ref = f"{PROJECT_ID}.{DATASET_ID}.{config['table_name']}"
    
    try:
        # Hitung tanggal 1 tahun terakhir
        lookback_date = (datetime.now() - timedelta(days=config['lookback_years']*365)).strftime('%Y-%m-%d')
        
        # QUERY TANPA LIMIT - AMBIL SEMUA DATA 1 TAHUN
        query = f"""
            SELECT datetime, close
            FROM `{table_ref}`
            WHERE datetime >= '{lookback_date}'
            ORDER BY datetime
            -- TANPA LIMIT: AMBIL SEMUA DATA 1 TAHUN
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
    
    # Handle missing values
    df['close'] = df['close'].ffill().bfill()  # Forward fill lalu backward fill
    
    print(f"   Data range: {df['datetime'].min()} to {df['datetime'].max()}")
    print(f"   Data points: {len(df):,}")
    print(f"   Last close price: {df['close'].iloc[-1]:.4f}")
    
    if len(df) < config['sequence_length'] * 3:
        print(f"   ✗ Data tidak cukup untuk training")
        return None
    
    # 3. NORMALISASI
    print(f"\n3. Normalisasi data...")
    close_prices = df['close'].values.reshape(-1, 1)
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    close_scaled = scaler.fit_transform(close_prices)
    print(f"   ✓ Data dinormalisasi")
    
    # 4. PREPARE SEQUENCES - SEMUA SEQUENCES TANPA STRIDE!
    print(f"\n4. Membuat sequences (SEMUA TANPA STRIDE)...")
    
    # Buat SEMUA sequences
    X, y = create_sequences(close_scaled, config['sequence_length'])
    
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    train_test_split_idx = int(len(df) * 0.8)
    
    print(f"   ✓ Sequences created (100% tanpa stride)")
    print(f"     Total data points: {len(df):,}")
    print(f"     Sequence length: {config['sequence_length']} (60 menit/1 jam)")
    print(f"     Total sequences: {len(X):,} (100% dari semua kemungkinan sequences)")
    print(f"     Train sequences: {len(X_train):,}")
    print(f"     Test sequences: {len(X_test):,}")
    
    # 5. TRAIN MODEL - 15 EPOCHS
    print(f"\n5. Training model LSTM (64 units, 15 epochs)...")
    model_path = f"/tmp/lstm_model_{timeframe_name}_all_data.h5"
    
    if force_retrain or not os.path.exists(model_path):
        print(f"   Training model baru dengan SEMUA DATA...")
        
        model = build_lstm_model(config['sequence_length'], units=config['lstm_units'])
        
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=config['patience'],
            restore_best_weights=True,
            verbose=1
        )
        
        model_checkpoint = ModelCheckpoint(
            model_path,
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
        
        print(f"   Training configuration:")
        print(f"     • Epochs: {config['max_epochs']}")
        print(f"     • Batch size: {config['batch_size']}")
        print(f"     • LSTM units: {config['lstm_units']}")
        print(f"     • Training samples: {len(X_train):,}")
        print(f"     • Validation samples: {len(X_test):,}")
        
        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=config['max_epochs'],
            batch_size=config['batch_size'],
            verbose=1,
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
    
    # Predict dengan batch besar
    y_pred_scaled = model.predict(X_test, batch_size=2048, verbose=0)
    y_pred = scaler.inverse_transform(y_pred_scaled)
    y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))
    
    mse = mean_squared_error(y_test_actual, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test_actual, y_pred)
    mape = mean_absolute_percentage_error(y_test_actual, y_pred) * 100
    accuracy = max(0, 100 - mape)
    
    print(f"   MSE:  {mse:.6f}")
    print(f"   RMSE: {rmse:.6f}")
    print(f"   MAE:  {mae:.6f}")
    print(f"   MAPE: {mape:.4f}%")
    print(f"   Akurasi: {accuracy:.2f}%")
    
    # 7. FORECASTING (15 MENIT)
    print(f"\n7. Forecasting {config['forecast_steps']} steps (15 menit)...")
    
    last_sequence = close_scaled[-config['sequence_length']:]
    forecasts = forecast_lstm(model, last_sequence, config['forecast_steps'], scaler)
    
    print(f"   ✓ Forecast generated untuk 15 menit ke depan:")
    
    last_actual = df['close'].iloc[-1]
    for i in range(len(forecasts)):
        forecast_price = forecasts[i]
        if last_actual > 0:
            pct_change = ((forecast_price - last_actual) / last_actual) * 100
            minutes = i + 1
            print(f"     {minutes} menit: {forecast_price:.4f} ({pct_change:+.2f}%)")
    
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
    
    success = save_data_all(
        timeframe_name=timeframe_name,
        df_actual=df,
        forecasts=forecasts,
        metrics=metrics_dict,
        train_test_split_idx=train_test_split_idx
    )
    
    if not success:
        return None
    
    # 9. FINAL SUMMARY
    print(f"\n" + "="*80)
    print("SUMMARY - 1 MENIT (FULL DATA VERSION)")
    print("="*80)
    
    summary_data = {
        'Data Period': f"1 tahun penuh (sejak {lookback_date})",
        'Total Data Points': f"{len(df):,}",
        'Sequence Length': f"{config['sequence_length']} (60 menit/1 jam)",
        'Forecast Horizon': f"{config['forecast_steps']} steps (15 menit)",
        'Model Units': f"{config['lstm_units']} LSTM units",
        'Training Epochs': f"{config['max_epochs']}",
        'Last Actual Price': f"{last_actual:.4f}",
        'Next Minute Forecast': f"{first_forecast:.4f}",
        '15-Minute Forecast': f"{forecasts[-1]:.4f}",
        'Change % (1min)': f"{pct_change:+.2f}%" if last_actual > 0 else "N/A",
        'MAPE': f"{mape:.4f}%",
        'Accuracy': f"{accuracy:.2f}%",
        'Training Sequences': f"{len(X_train):,} (100% tanpa stride)",
        'Testing Sequences': f"{len(X_test):,} (100% tanpa stride)",
        'Data Storage': f"SEMUA DATA disimpan ({len(df):,} rows)",
        'Processing Strategy': "FULL DATA - NO STRIDE - 15 EPOCHS"
    }
    
    summary_df = pd.DataFrame(list(summary_data.items()), columns=['Metric', 'Value'])
    print(summary_df.to_string(index=False))
    
    return {
        'timeframe': timeframe_name,
        'forecasts': forecasts,
        'metrics': metrics_dict,
        'last_price': df['close'].iloc[-1],
        'first_forecast': forecasts[0] if len(forecasts) > 0 else None,
        'total_data_points': len(df),
        'total_sequences': len(X),
        'training_sequences': len(X_train),
        'testing_sequences': len(X_test)
    }

# ==================== JALANKAN ====================
if __name__ == "__main__":
    print("\n" + "="*80)
    print("MEMULAI PREDIKSI LSTM UNTUK 1 MENIT")
    print("KONFIGURASI:")
    print("  • Data: 1 tahun penuh (tanpa limit)")
    print("  • Model: 64 LSTM units")
    print("  • Forecast: 15 menit ke depan")
    print("  • Sequence length: 60 (1 jam history)")
    print("  • Training: SEMUA DATA tanpa stride")
    print("  • Epochs: 15 epochs")
    print("="*80 + "\n")
    
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
    result = process_1menit_all_data(force_retrain=True)
    
    if result:
        print("\n" + "="*80)
        print("PREDIKSI 1 MENIT SELESAI")
        print("="*80)
        print(f"✓ SEMUA DATA 1 TAHUN telah diproses")
        print(f"✓ Tabel BigQuery: {PROJECT_ID}.{PREDICTION_DATASET}.lstm_1menit")
        print(f"✓ Total data points: {result['total_data_points']:,}")
        print(f"✓ Total sequences: {result['total_sequences']:,} (100% tanpa stride)")
        print(f"✓ Training sequences: {result['training_sequences']:,}")
        print(f"✓ Testing sequences: {result['testing_sequences']:,}")
        print(f"✓ Forecast 1 menit: {result['first_forecast']:.4f}")
        print(f"✓ Forecast 15 menit: {result['forecasts'][-1]:.4f}")
        print(f"✓ Model accuracy: {result['metrics']['accuracy']:.2f}%")
        print(f"✓ Data storage: SEMUA data disimpan ke BigQuery")
    else:
        print("\n" + "="*80)
        print("PREDIKSI 1 MENIT GAGAL")
        print("="*80)