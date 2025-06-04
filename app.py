from fastapi import FastAPI, HTTPException
from typing import Dict
import numpy as np
import pickle
import yfinance as yf
import onnxruntime as ort

app = FastAPI()

with open("model/scalers.pkl", "rb") as f:
    scalers = pickle.load(f)

# Load ONNX model session once on startup
session = ort.InferenceSession("model/model_lstm.onnx")
input_name = session.get_inputs()[0].name

window_size = 30  # Match your model input shape


forecast_steps = {
    '1_day': 1,
    '1_week': 7,
    '1_month': 30,
    '3_months': 90,
    '6_months': 180
}

symbols = ['BBCA', 'BYAN', 'TPIA', 'BBRI', 'BMRI', 'DSSA', 'TLKM', 'ASII', 'BBNI', 'ICBP']

def fetch_last_60_days(symbol: str) -> np.ndarray:
    yf_symbol = symbol + ".JK"
    df = yf.Ticker(yf_symbol).history(period="30d", interval="1d")
    if df.empty or 'Close' not in df:
        raise ValueError(f"No data found for {symbol}")
    print(f"Fetched and cached new data for {symbol}")
    return df['Close'].values

def predict_with_onnx(input_seq: np.ndarray) -> float:
    # ONNX expects input as a dict
    input_name = session.get_inputs()[0].name
    output = session.run(None, {input_name: input_seq})[0]
    return output[0, 0]

def forecast_symbol(symbol: str) -> Dict[str, float]:
    close_series = fetch_last_60_days(symbol)
    scaler = scalers.get(symbol)
    if scaler is None:
        raise ValueError(f"Scaler not found for symbol {symbol}")

    result = {}
    for label, steps in forecast_steps.items():
        preds_scaled = predict_future_with_onnx(close_series, steps, window_size, scaler)
        preds_original = scaler.inverse_transform(np.array(preds_scaled).reshape(-1, 1)).flatten()
        result[label] = float(preds_original[-1])

    return result

def forecast_all_symbols() -> Dict[str, Dict[str, float]]:
    forecasts = {}

    for symbol in symbols:
        clean_symbol = symbol.replace(".JK", "")
        try:
            close_series = fetch_last_60_days(symbol)
            scaler = scalers[clean_symbol]

            result = {}
            for label, steps in forecast_steps.items():
                preds_scaled = predict_future_with_onnx(close_series, steps, window_size, scaler)
                preds_original = scaler.inverse_transform(np.array(preds_scaled).reshape(-1, 1)).flatten()
                result[label] = float(preds_original[-1])

            forecasts[clean_symbol] = result

        except Exception as e:
            forecasts[clean_symbol] = {"error": str(e)}

    return forecasts

def predict_future_with_onnx(series, steps_ahead, window_size, scaler):
    last_window = series[-window_size:].reshape(-1, 1)
    input_seq = scaler.transform(last_window).reshape(1, window_size, 1).astype(np.float32)

    preds = []
    for _ in range(steps_ahead):
        pred = predict_with_onnx(input_seq)
        preds.append(pred)
        input_seq = np.append(input_seq[:, 1:, :], [[[pred]]], axis=1).astype(np.float32)
    return preds

@app.get("/")
def home():
    return {"message": "Live Stock Forecast API using yfinance"}


@app.get("/forecast")
def get_all_forecast():
    forecasts = {}
    for symbol in symbols:
        try:
            forecasts[symbol] = forecast_symbol(symbol)
        except Exception as e:
            forecasts[symbol] = {"error": str(e)}
    return forecasts

@app.get("/forecast/{symbol}")
def get_single_forecast(symbol: str):
    symbol = symbol.upper()
    if symbol not in symbols:
        raise HTTPException(status_code=404, detail=f"Symbol '{symbol}' not supported.")
    try:
        result = forecast_symbol(symbol)
        return {symbol: result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
