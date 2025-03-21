import pandas as pd
from rapidfuzz import fuzz


def name_similarity(a, b):
    if pd.isnull(a) or pd.isnull(b):
        return 0
    return fuzz.token_sort_ratio(a, b) / 100


def address_similarity(a, b):
    if pd.isnull(a) or pd.isnull(b):
        return 0
    sort_sim = fuzz.token_sort_ratio(a, b)
    set_sim = fuzz.token_set_ratio(a, b)
    return max(sort_sim, set_sim) / 100


def calculate_weighted_similarity(name_sim, address_sim, domain_sim):
    return (name_sim * 0.6) + (address_sim * 0.3) + (domain_sim * 0.1)


def calculate_similarities(df):
    results = []

    grouped = df.groupby(["main_country", "main_city"])
    for (country, city), subset_df in grouped:
        print(f"Calculating similarities in {country}, {city} ({len(subset_df)} companies)...")

        for i in range(len(subset_df)):
            for j in range(i + 1, len(subset_df)):
                name_sim = name_similarity(subset_df.iloc[i]["company_name"], subset_df.iloc[j]["company_name"])
                address_sim = address_similarity(subset_df.iloc[i]["address"], subset_df.iloc[j]["address"])
                domain_sim = name_similarity(subset_df.iloc[i]["domains"], subset_df.iloc[j]["domains"])
                weighted_sim = calculate_weighted_similarity(name_sim, address_sim, domain_sim)

                results.append({
                    "company_1": subset_df.iloc[i]["company_name"],
                    "company_2": subset_df.iloc[j]["company_name"],
                    "main_country": country,
                    "main_city": city,
                    "name_similarity": name_sim,
                    "address_similarity": address_sim,
                    "website_similarity": domain_sim,
                    "weighted_similarity": weighted_sim,
                })

    return pd.DataFrame(results)


def find_similar_companies(df, similarity_file):
    result_df = calculate_similarities(df)
    result_df.to_parquet(similarity_file, index=False)
    print(f"Similarities saved to {similarity_file}")
