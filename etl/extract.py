import yfinance as yf
import pandas as pd
import datetime 

def fetch_close_data():
    tickers = ['BBCA.JK', 'BYAN.JK', 'TPIA.JK', 'BBRI.JK', 'BMRI.JK',
               'DSSA.JK', 'TLKM.JK', 'ASII.JK', 'BBNI.JK', 'ICBP.JK']

    end_date = datetime.datetime.today().strftime('%Y-%m-%d')
    stock_close_datas = {}

    for ticker in tickers:
        df = yf.download(ticker, start='2018-01-01', end=end_date, interval='1d')
        stock_close_datas[ticker] = df['Close']
        
    df_all = pd.concat(stock_close_datas, axis=1)
    df_all.columns = tickers

    return df_all