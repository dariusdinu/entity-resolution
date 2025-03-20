from constants import CLEANED_FILE, SIMILARITY_FILE
from preprocessing import load_data, preprocess_data
from similarity import find_similar_companies
from src.random_forest_clustering import group_similar_companies_rf
from src.rule_based_clustering import group_similar_companies
from src.xgboost_clustering import group_similar_companies_XGBoost


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
