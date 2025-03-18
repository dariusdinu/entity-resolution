import re
import pandas as pd
from src.utils.name_cleaning_utils import clean_text


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
