import pandas as pd
from rapidfuzz import fuzz


def levenshtein_similarity(str1, str2):
    if pd.isnull(str1) or pd.isnull(str2):
        return 0
    return fuzz.partial_ratio(str1, str2) / 100


def name_similarity(a, b):
    return fuzz.token_sort_ratio(a, b) / 100


def calculate_weighted_similarity(name_sim, address_sim, website_sim):
    return (name_sim * 0.6) + (address_sim * 0.3) + (website_sim * 0.1)


def calculate_similarities(df):
    results = []

    grouped = df.groupby(["main_country", "main_city"])
    for (country, city), subset_df in grouped:
        print(f"Calculating similarities in {country}, {city} ({len(subset_df)} companies)...")

        for i in range(len(subset_df)):
            for j in range(i + 1, len(subset_df)):
                name_sim = name_similarity(subset_df.iloc[i]["company_name"], subset_df.iloc[j]["company_name"])
                address_sim = name_similarity(subset_df.iloc[i]["address"], subset_df.iloc[j]["address"])
                website_sim = name_similarity(subset_df.iloc[i]["domains"], subset_df.iloc[j]["domains"])
                weighted_sim = calculate_weighted_similarity(name_sim, address_sim, website_sim)

                results.append({
                    "company_1": subset_df.iloc[i]["company_name"],
                    "company_2": subset_df.iloc[j]["company_name"],
                    "main_country": country,
                    "main_city": city,
                    "name_similarity": name_sim,
                    "address_similarity": address_sim,
                    "website_similarity": website_sim,
                    "weighted_similarity": weighted_sim,
                })

    return pd.DataFrame(results)


def find_similar_companies(df, similarity_file):
    result_df = calculate_similarities(df)
    result_df.to_parquet(similarity_file, index=False)
    print(f"Similarities saved to {similarity_file}")
