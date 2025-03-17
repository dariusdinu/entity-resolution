import re
import pandas as pd
from utils.country_utils import remove_exact_duplicates, standardize_countries
from utils.general_utils import clean_text, handle_missing_values
from utils.enrichment_utils import enrich_name, enrich_address, enrich_domain

INITIAL_FILE = "../data/raw_companies.parquet"
CLEANED_FILE = "../data/cleaned_companies.parquet"


def load_data(file_path):
    return pd.read_parquet(file_path)


def enrich_dataset(df):
    """Applies fallback enrichment functions from utils."""
    df["company_name"] = df.apply(enrich_name, axis=1)
    df["address"] = df.apply(enrich_address, axis=1)
    df["domains"] = df["domains"].apply(enrich_domain)
    df["all_domains"] = df["all_domains"].apply(enrich_domain)
    return df


def clean_company_names(df):
    df["company_name"] = df["company_name"].apply(clean_text)
    return df


def clean_websites(df):
    def normalize_domain(domain):
        if pd.isnull(domain) or domain == '\\N':
            return None
        domain = domain.lower().strip()
        domain = re.sub(r'^www\.', '', domain)
        domain = domain.split('/')[0]
        return domain

    df["domains"] = df["domains"].apply(normalize_domain)
    df["all_domains"] = df["all_domains"].apply(normalize_domain)
    return df


def clean_addresses(df):
    df["address"] = df["address"].apply(clean_text)
    return df


def save_cleaned_data(df, file_path):
    df.to_parquet(file_path, index=False)


def preprocess_data():
    df = load_data(INITIAL_FILE)
    df = handle_missing_values(df)
    df = enrich_dataset(df)
    df = clean_company_names(df)
    df = clean_websites(df)
    df = clean_addresses(df)
    df = remove_exact_duplicates(df)
    df = standardize_countries(df)

    save_cleaned_data(df, CLEANED_FILE)
    return df
