from clustering_rule_based import group_similar_companies
from clustering_XGBoost import group_similar_companies_ml as ml_clustering
from data_processing import load_data, preprocess_data
from insights import run_insights
from src.similarity import find_similar_companies
from clustering_RF import group_similar_companies_rf

# File paths
INITIAL_FILE = "../data/raw_companies.parquet"
CLEANED_FILE = "../data/cleaned_companies.parquet"
SIMILARITY_FILE = "../data/similarity_results.parquet"


def main():
    # STEP 1: Optional Preprocessing
    # df = preprocess_data()

    # STEP 2: Load enriched dataset
    # df = load_data(CLEANED_FILE)

    # STEP 3: Optional Similarity Calculation
    # final_similarity_df = find_similar_companies(df)
    # final_similarity_df.to_parquet(SIMILARITY_FILE, index=False)

    # ----- STEP 3a: Rule-Based Clustering -----
    group_similar_companies()
    run_insights("../data/grouped_companies_rule_based.parquet", "../data/sample_groups_rule_based.csv")

    # ----- STEP 3b: Machine Learning-based Clustering (XGBoost) -----
    ml_clustering()
    run_insights("../data/grouped_companies_xgboost.parquet", "../data/sample_groups_xgboost.csv")

    # ----- STEP 3c: Machine Learning-based Clustering (Random Forests) -----
    group_similar_companies_rf()
    run_insights("../data/grouped_companies_rf.parquet", "../data/sample_groups_rf.csv")


if __name__ == "__main__":
    main()
