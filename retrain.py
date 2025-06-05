import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import os

def load_and_preprocess(filepath, window_size=30):
    df = pd.read_csv(filepath, parse_dates=["Date"])
    df = df.drop(columns=["Date"])  

    scalers = {}
    standardized_stocks = {}

    for symbol in df.columns:
        scaler = StandardScaler()
        series = df[symbol].values.reshape(-1, 1)
        standardized = scaler.fit_transform(series)
        scalers[symbol] = scaler
        standardized_stocks[symbol] = standardized.squeeze()

    def split_dataset(series):
        X, y = [], []
        for i in range(len(series) - window_size):
            X.append(series[i:i + window_size])
            y.append(series[i + window_size])
        return np.array(X).reshape(-1, window_size, 1), np.array(y)

    X_all, y_all = [], []
    for series in standardized_stocks.values():
        X, y = split_dataset(series)
        X_all.append(X)
        y_all.append(y)

    X_all = np.concatenate(X_all)
    y_all = np.concatenate(y_all)

    # Shuffle
    indices = np.arange(len(X_all))
    np.random.shuffle(indices)
    X_all, y_all = X_all[indices], y_all[indices]

    return X_all, y_all

def retrain_existing_model(csv_path='./stocks.csv', model_path='./model/model_lstm.h5'):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    X, y = load_and_preprocess(csv_path)

    print(f"Loaded data shape: {X.shape}, {y.shape}")
    model = load_model(model_path)

    print("Retraining model...")
    model.fit(X, y, epochs=5, batch_size=32, verbose=1)
    model.save(model_path)
    print(f"Updated model saved to {model_path}")