import pandas as pd
from rapidfuzz import fuzz

SIMILARITY_FILE = "../data/similarity_results.parquet"


def levenshtein_similarity(str1, str2):
    if pd.isnull(str1) or pd.isnull(str2):
        return 0
    return fuzz.partial_ratio(str1, str2) / 100


def name_similarity(a, b):
    return fuzz.token_sort_ratio(a, b) / 100  # Normalize to 0-1


def calculate_similarities(df):
    similarities = []

    total_comparisons = len(df) * (len(df) - 1) // 2  # Total comparisons to be made

    for i in range(len(df)):
        for j in range(i + 1, len(df)):  # Compare every company pair
            name_sim = levenshtein_similarity(df.iloc[i]["company_name"], df.iloc[j]["company_name"])
            address_sim = name_similarity(df.iloc[i]["main_street"], df.iloc[j]["main_street"])

            similarities.append({
                "company_1": df.iloc[i]["company_name"],
                "company_2": df.iloc[j]["company_name"],
                "name_similarity": name_sim,
                "address_similarity": address_sim
            })

    return pd.DataFrame(similarities)


def calculate_weighted_similarity(name_sim, address_sim, website_sim):
    return (name_sim * 0.7) + (address_sim * 0.2) + (website_sim * 0.1)


def find_similar_companies(df):
    results = []

    grouped = df.groupby("main_country")

    for country, subset_df in grouped:
        for i in range(len(subset_df)):
            for j in range(i + 1, len(subset_df)):
                name_sim = name_similarity(subset_df.iloc[i]["company_name"], subset_df.iloc[j]["company_name"])
                address_sim = name_similarity(subset_df.iloc[i]["main_street"], subset_df.iloc[j]["main_street"])
                website_sim = name_similarity(subset_df.iloc[i]["domains"], subset_df.iloc[j]["domains"])

                weighted_sim = calculate_weighted_similarity(name_sim, address_sim, website_sim)

                results.append({
                    "company_1": subset_df.iloc[i]["company_name"],
                    "company_2": subset_df.iloc[j]["company_name"],
                    "main_country": country,
                    "name_similarity": name_sim,
                    "address_similarity": address_sim,
                    "website_similarity": website_sim,
                    "weighted_similarity": weighted_sim,
                })

    print("Similarity calculation completed!")
    return pd.DataFrame(results)
