from data_processing import (
    load_data, handle_missing_values, clean_company_names,
    clean_websites, clean_addresses, save_cleaned_data
)

INITIAL_FILE = "../data/raw_companies.parquet"
CLEANED_FILE = "../data/cleaned_companies.parquet"


def main():
    df = load_data(INITIAL_FILE)

    df = handle_missing_values(df)
    df = clean_company_names(df)
    df = clean_websites(df)
    df = clean_addresses(df)

    save_cleaned_data(df, CLEANED_FILE)

    print("Data cleaning complete!")

if __name__ == "__main__":
    main()
