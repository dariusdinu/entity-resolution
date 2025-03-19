import re

import pandas as pd


def clean_text(text):
    if pd.isnull(text):
        return None
    text = text.lower().strip()
    text = re.sub(r'[^a-zA-Z0-9 ]', '', text)  # Special characters removal
    return text
