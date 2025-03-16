from clustering import group_similar_companies
from data_processing import load_data, preprocess_data
from similarity import find_similar_companies

# File paths
INITIAL_FILE = "../data/raw_companies.parquet"
CLEANED_FILE = "../data/cleaned_companies.parquet"
SIMILARITY_FILE = "../data/similarity_results.parquet"
GROUPED_FILE_PARQUET = "../data/grouped_companies.parquet"
GROUPED_FILE_CSV = "../data/grouped_companies.csv"


def main():
    df = preprocess_data()

    df = load_data(CLEANED_FILE)

    final_similarity_df = find_similar_companies(df)

    final_similarity_df.to_parquet(SIMILARITY_FILE, index=False)
    print(f"Saved the similarity results to local file")

    # STEP 5: CLUSTER SIMILAR COMPANIES
    group_similar_companies()

    print("\nCompleted the clustering process")


if __name__ == "__main__":
    main()
