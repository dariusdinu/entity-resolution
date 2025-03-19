import pandas as pd

from src.utils.data_cleaning import clean_company_names, clean_websites, clean_addresses
from utils.country import standardize_countries
from utils.enrichment import enrich_name, enrich_address, enrich_domain

INITIAL_FILE = "../data/raw_companies.parquet"
CLEANED_FILE = "../data/cleaned_companies.parquet"


def load_data(file_path):
    return pd.read_parquet(file_path)


def enrich_dataset(df):
    df["company_name"] = df.apply(enrich_name, axis=1)
    df["address"] = df.apply(enrich_address, axis=1)
    df["domains"] = df["domains"].apply(enrich_domain)
    df["all_domains"] = df["all_domains"].apply(enrich_domain)
    return df


def save_cleaned_data(df, file_path):
    df.to_parquet(file_path, index=False)


def handle_missing_values(df):
    df.replace({'\\N': None, '': None}, inplace=True)
    return df


def preprocess_data():
    df = load_data(INITIAL_FILE)
    df = handle_missing_values(df)
    df = enrich_dataset(df)
    df = clean_company_names(df)
    df = clean_websites(df)
    df = clean_addresses(df)
    df = standardize_countries(df)

    save_cleaned_data(df, CLEANED_FILE)
    return df
