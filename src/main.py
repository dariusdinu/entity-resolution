from data_processing import (
    load_data, clean_company_names, clean_websites, clean_addresses,
    save_cleaned_data
)
from utils.general_utils import handle_missing_values
from utils.country_utils import remove_exact_duplicates, standardize_countries

# File paths
INITIAL_FILE = "../data/raw_companies.parquet"
CLEANED_FILE = "../data/cleaned_companies.parquet"

def main():
    df = load_data(INITIAL_FILE)

    df = handle_missing_values(df)
    df = clean_company_names(df)
    df = clean_websites(df)
    df = clean_addresses(df)

    df = remove_exact_duplicates(df)

    df = standardize_countries(df)

    save_cleaned_data(df, CLEANED_FILE)

if __name__ == "__main__":
    main()

