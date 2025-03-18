from clustering_rule_based import group_similar_companies
from clustering_XGBoost import group_similar_companies_XGBoost
from data_processing import load_data, preprocess_data
from similarity import find_similar_companies
from clustering_RF import group_similar_companies_rf

INITIAL_FILE = "../data/raw_companies.parquet"
CLEANED_FILE = "../data/cleaned_companies.parquet"
SIMILARITY_FILE = "../data/similarity_results.parquet"


def main():
    # STEP 1: Preprocessing
    df = preprocess_data()

    # STEP 2: Load enriched dataset
    df = load_data(CLEANED_FILE)

    # STEP 3: Similarity Calculation
    find_similar_companies(df, SIMILARITY_FILE)

    # STEP 4a: Rule-Based Clustering
    group_similar_companies(SIMILARITY_FILE, CLEANED_FILE)

    # STEP 4b: Machine Learning-based Clustering (XGBoost)
    group_similar_companies_XGBoost(SIMILARITY_FILE, CLEANED_FILE)

    # STEP 4c: Machine Learning-based Clustering (Random Forests)
    group_similar_companies_rf(SIMILARITY_FILE, CLEANED_FILE)


if __name__ == "__main__":
    main()
