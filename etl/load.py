import pandas as pd
import os

def save_to_csv(df: pd.DataFrame, filepath: str):

    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    df.to_csv(filepath, index=False)
    print(f"Data saved to {filepath}")