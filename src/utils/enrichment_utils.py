import pandas as pd


def enrich_name(row):
    if pd.notnull(row["company_name"]):
        return row["company_name"]
    for col in ["company_legal_names", "company_commercial_names"]:
        if pd.notnull(row[col]):
            return row[col].split("|")[0].strip()
    return None


def enrich_address(row):
    if pd.notnull(row["main_address_raw_text"]):
        return row["main_address_raw_text"]
    elif pd.notnull(row["main_street"]):
        return row["main_street"]
    return None


def enrich_domain(val):
    if pd.notnull(val):
        return val.split("|")[0].strip()
    return None
