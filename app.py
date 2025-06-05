from fastapi import FastAPI, HTTPException
from model_utils import make_prediction
import pandas as pd

app = FastAPI()

@app.get("/")
def index():
    return {
        "message": "Welcome to the Stock Predictor API",
        "usage": "/predict?symbol=BBCA"
    }

@app.get("/predict")
def predict(symbol: str):
    try:
        result = make_prediction(symbol.upper())
        return {"symbol": symbol.upper(), "predictions": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/latest-stock")
def get_latest_stock():
    try:
        stocks = pd.read_csv("stocks.csv")
        latest_stock = stocks.tail(1).iloc[0]
        return latest_stock.to_dict()
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="stocks.csv not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))