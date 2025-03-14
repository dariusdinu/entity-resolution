from data_processing import preprocess_data, load_data
from similarity import find_similar_companies

INITIAL_FILE = "../data/raw_companies.parquet"
CLEANED_FILE = "../data/cleaned_companies.parquet"
SIMILARITY_FILE = "../data/similarity_results.parquet"


def main():
    # PREPROCESSING
    # df = preprocess_data()

    # LOAD CLEANED DATA
    print("Loading cleaned dataset...")
    df = load_data(CLEANED_FILE)

    # FIND SIMILAR COMPANIES
    final_similarity_df = find_similar_companies(df)

    # SAVE SIMILARITY RESULTS
    print(f"Saving similarity data for next step: {SIMILARITY_FILE}")
    final_similarity_df.to_parquet(SIMILARITY_FILE, index=False)


if __name__ == "__main__":
    main()
