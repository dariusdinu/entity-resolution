import pandas as pd
import re

INITIAL_FILE = "../data/raw_companies.parquet"
CLEANED_FILE = "../data/cleaned_companies.parquet"

def load_data(file_path):
    return pd.read_parquet(file_path)


def clean_text(text):
    if pd.isnull(text):
        return None
    text = text.lower().strip()
    text = re.sub(r'[^a-zA-Z0-9 ]', '', text)
    return text


def clean_company_names(df):
    suffixes = [" llc", " inc", " ltd", " co", " corporation", " corp", " gmbh", " srl"]

    def remove_suffix(name):
        if pd.isnull(name):
            return None
        name = clean_text(name)
        for suffix in suffixes:
            if name.endswith(suffix):
                name = name.replace(suffix, '')
        return name.strip()

    df["company_name"] = df["company_name"].apply(remove_suffix)
    df["company_legal_names"] = df["company_legal_names"].apply(remove_suffix)
    df["company_commercial_names"] = df["company_commercial_names"].apply(remove_suffix)
    return df


def clean_websites(df):
    def normalize_domain(domain):
        if pd.isnull(domain) or domain == '\\N':
            return None
        domain = domain.lower().strip()
        domain = re.sub(r'^www\.', '', domain)  # Remove www.
        domain = domain.split('/')[0]  # Remove URL paths
        return domain

    df["domains"] = df["domains"].apply(normalize_domain)
    df["all_domains"] = df["all_domains"].apply(normalize_domain)
    return df


def clean_addresses(df):
    address_columns = ["main_street", "main_city", "main_postcode"]
    for col in address_columns:
        df[col] = df[col].apply(clean_text)
    return df


def handle_missing_values(df):
    df.replace({'\\N': None, '': None}, inplace=True)
    return df


def save_cleaned_data(df, file_path):
    df.to_parquet(file_path, index=False)
    print(f"Cleaned data saved to {file_path}")
