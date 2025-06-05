import pandas as pd

def preprocess_data(df):

    df_ffil = df.ffill()

    full_date_range = pd.date_range(start=df_ffil.index.min(), end=df_ffil.index.max(), freq='D')
    df_ffil = df_ffil.reindex(full_date_range).ffill()
    df_ffil.index.name = 'Date'

    df_ffil.columns = df_ffil.columns.str.replace('.JK', '', regex=False)

    df_ffil = df_ffil.reset_index()

    df_ffil = df_ffil.drop_duplicates()

    df_ffil = df_ffil.sort_values(by='Date', ascending=True)

    return df_ffil