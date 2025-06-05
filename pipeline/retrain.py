import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.metrics import MeanAbsoluteError, MeanAbsolutePercentageError
import tf2onnx
import onnx
import tensorflow as tf
import joblib

# --- Configuration ---
TICKERS = ['ASII', 'BBCA', 'BBNI', 'BBRI', 'BMRI', 'BYAN', 'DSSA', 'ICBP', 'TLKM', 'TPIA']
DATA_PATH = './stocks.csv'
MODEL_DIR = './models'
EXPORT_ONNX_DIR = './onnx_models'
SCALER_DIR = './scalers'
WINDOW_SIZE = 30
HORIZON = 1
TRAIN_RATIO = 0.7
EVAL_RATIO = 0.15

def create_sliding_window(data, window_size=30, horizon=1):
    X, y = [], []
    for i in range(len(data) - window_size - horizon + 1):
        X.append(data[i:i + window_size])
        y.append(data[i + window_size + horizon - 1])
    return np.array(X), np.array(y)

def split_data(data, window_size, horizon):
    X, y = create_sliding_window(data, window_size, horizon)
    n = len(X)
    train_end = int(n * TRAIN_RATIO)
    eval_end = train_end + int(n * EVAL_RATIO)

    return (X[:train_end], y[:train_end],
            X[train_end:eval_end], y[train_end:eval_end],
            X[eval_end:], y[eval_end:])

def retrain_model(ticker, df):
    print(f"\nRetraining model for {ticker}...")

    # Load existing scaler
    scaler_path = os.path.join(SCALER_DIR, f"{ticker}_scaler.pkl")
    if not os.path.exists(scaler_path):
        print(f"Scaler for {ticker} not found at {scaler_path}. Skipping...")
        return
    scaler = joblib.load(scaler_path)

    # Preprocessing with loaded scaler
    scaled_data = scaler.transform(df[[ticker]]).squeeze()

    X_train, y_train, X_val, y_val, _, _ = split_data(scaled_data, WINDOW_SIZE, HORIZON)
    X_train = X_train.reshape(-1, WINDOW_SIZE, 1)
    X_val = X_val.reshape(-1, WINDOW_SIZE, 1)

    # Load existing model (.h5)
    model_path = os.path.join(MODEL_DIR, f"{ticker}_model.h5")
    if not os.path.exists(model_path):
        print(f"Model for {ticker} not found at {model_path}. Skipping...")
        return
    
    tf.config.run_functions_eagerly(True)

    custom_objects = {
    'mae': MeanAbsoluteError(),
    'mape': MeanAbsolutePercentageError()
    }

    model = load_model(model_path, custom_objects=custom_objects)
    model.compile(optimizer='adam', loss='mae', metrics=['mape'])

    # Retrain
    early_stop = EarlyStopping(patience=5, restore_best_weights=True)
    model.fit(X_train, y_train,
              validation_data=(X_val, y_val),
              epochs=50,
              batch_size=32,
              callbacks=[early_stop],
              verbose=1)

    # Save updated .h5 model
    model.save(model_path)
    print(f"Saved updated model to {model_path}")

    # Simpan model dalam format onnx
    model.output_names = ['output']
    # Mendapatkan input shape dari model
    input_shape = model.input_shape[1:] # Tidak Mengambil dimensi batch

    # Menetapkan input signature
    input_signature = [tf.TensorSpec(shape=[None] + list(input_shape), dtype=tf.float64, name='input')]

    # Konversi model Keras ke ONNX
    onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature=input_signature)

    # Simpan model ONNX
    onnx.save_model(onnx_model, f"models/{ticker}_model.onnx")

def main():
    os.makedirs(EXPORT_ONNX_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    df = pd.read_csv(DATA_PATH, parse_dates=['Date'])
    df.set_index('Date', inplace=True)

    for ticker in TICKERS:
        if ticker not in df.columns:
            print(f"Ticker {ticker} not found in dataset. Skipping...")
            continue
        retrain_model(ticker, df)

if __name__ == "__main__":
    main()
