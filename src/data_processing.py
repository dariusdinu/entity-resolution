import re
import pandas as pd
from utils.country_utils import remove_exact_duplicates, standardize_countries
from utils.general_utils import clean_text, handle_missing_values

INITIAL_FILE = "../data/raw_companies.parquet"
CLEANED_FILE = "../data/cleaned_companies.parquet"


def load_data(file_path):
    return pd.read_parquet(file_path)


def enrich_dataset(df):
    def enrich_name(row):
        if pd.notnull(row["company_name"]):
            return row["company_name"]
        for col in ["company_legal_names", "company_commercial_names"]:
            if pd.notnull(row[col]):
                return row[col].split("|")[0].strip()
        return None

    df["company_name"] = df.apply(enrich_name, axis=1)

    def enrich_address(row):
        if pd.notnull(row["main_address_raw_text"]):
            return row["main_address_raw_text"]
        elif pd.notnull(row["main_street"]):
            return row["main_street"]
        return None

    df["address"] = df.apply(enrich_address, axis=1)

    def enrich_domain(val):
        if pd.notnull(val):
            return val.split("|")[0].strip()
        return None

    df["domains"] = df["domains"].apply(enrich_domain)
    df["all_domains"] = df["all_domains"].apply(enrich_domain)

    return df


def clean_company_names(df):
    suffixes = [
        " llc", " inc", " ltd", " co", " corporation", " corp", " gmbh", " srl", " pty",
        " ag", " limited", " holdings", " enterprises", " solutions", " group", " systems"
    ]

    def remove_suffix(name):
        if pd.isnull(name):
            return None
        name = clean_text(name)
        for suffix in suffixes:
            if name.endswith(suffix):
                name = name.replace(suffix, '')
        return name.strip()

    df["company_name"] = df["company_name"].apply(remove_suffix)
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
