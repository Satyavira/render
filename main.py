from etl.extract import fetch_close_data
from etl.transform import preprocess_data
from etl.load import save_to_csv

df_raw = fetch_close_data()
df_clean = preprocess_data(df_raw)
save_to_csv(df_clean, "./fincoach/stocks.csv")