from pipeline.extract import fetch_close_data
from pipeline.transform import preprocess_data
from pipeline.load import save_to_csv
from pipeline import retrain

def main():
    df_raw = fetch_close_data()

    df_clean = preprocess_data(df_raw)

    csv_path = "./stocks.csv"
    save_to_csv(df_clean, csv_path)

    # Retrain
    retrain.main()

if __name__ == "__main__":
    main()
