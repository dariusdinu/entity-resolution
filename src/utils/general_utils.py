import re

import pandas as pd


def clean_text(text):
    if pd.isnull(text):
        return None
    text = text.lower().strip()
    text = re.sub(r'[^a-zA-Z0-9 ]', '', text)  # Remove special characters
    return text


def handle_missing_values(df):
    df.replace({'\\N': None, '': None}, inplace=True)
    return df
