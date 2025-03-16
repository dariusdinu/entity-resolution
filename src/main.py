from clustering_rule_based import group_similar_companies
from data_processing import load_data
from insights import run_insights

# File paths
INITIAL_FILE = "../data/raw_companies.parquet"
CLEANED_FILE = "../data/cleaned_companies.parquet"
SIMILARITY_FILE = "../data/similarity_results.parquet"
GROUPED_FILE_PARQUET = "../data/grouped_companies.parquet"
GROUPED_FILE_CSV = "../data/grouped_companies.csv"


def main():
    # STEP 1: Optional Preprocessing
    # df = preprocess_data()

    # STEP 2: Load enriched dataset
    df = load_data(CLEANED_FILE)

    # STEP 3: Optional Similarity Calculation
    # final_similarity_df = find_similar_companies(df)
    # final_similarity_df.to_parquet(SIMILARITY_FILE, index=False)

    # STEP 4: Clustering
    # group_similar_companies()

    # STEP 5: Interpretation / Analysis
    run_insights()

if __name__ == "__main__":
    main()
