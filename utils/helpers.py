import pandas

# Saver for preprocessed dataframe
def save_df(df, set):
    df.to_csv(f'/work/preprocessed_{set}.csv', sep=',', encoding='utf-8', index=False)

# Loader for preprocessed dataframe
def load_df(path):
    df = pandas.read_csv(path)
    return df