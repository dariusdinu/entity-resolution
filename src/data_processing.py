import pandas as pd
import re
from utils.general_utils import clean_text, handle_missing_values
from utils.country_utils import remove_exact_duplicates, standardize_countries

# File paths
INITIAL_FILE = "../data/raw_companies.parquet"
CLEANED_FILE = "../data/cleaned_companies.parquet"


def load_data(file_path):
    return pd.read_parquet(file_path)


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


def save_cleaned_data(df, file_path):
    df.to_parquet(file_path, index=False)


def preprocess_data():
    df = load_data(INITIAL_FILE)
    df = handle_missing_values(df)
    df = clean_company_names(df)
    df = clean_websites(df)
    df = clean_addresses(df)
    df = remove_exact_duplicates(df)
    df = standardize_countries(df)

    save_cleaned_data(df, CLEANED_FILE)
    return df
