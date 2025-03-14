import os
import pandas as pd
from rapidfuzz import fuzz

SIMILARITY_FILE = "../data/similarity_results.parquet"


def levenshtein_similarity(str1, str2):
    if pd.isnull(str1) or pd.isnull(str2):
        return 0
    return fuzz.partial_ratio(str1, str2) / 100


def jaccard_similarity(str1, str2):
    if pd.isnull(str1) or pd.isnull(str2) or str1.strip() == "" or str2.strip() == "":
        return 0

    set1, set2 = set(str1.split()), set(str2.split())
    union_size = len(set1 | set2)

    if union_size == 0:
        return 0

    return len(set1 & set2) / union_size


def calculate_similarities(df):
    similarities = []

    for i in range(len(df)):
        for j in range(i + 1, len(df)):
            name_sim = levenshtein_similarity(df.iloc[i]["company_name"], df.iloc[j]["company_name"])
            address_sim = jaccard_similarity(df.iloc[i]["main_street"], df.iloc[j]["main_street"])

            similarities.append({
                "company_1": df.iloc[i]["company_name"],
                "company_2": df.iloc[j]["company_name"],
                "name_similarity": name_sim,
                "address_similarity": address_sim
            })

    return pd.DataFrame(similarities)


def find_similar_companies(df, force_recompute=False, top_n=10):
    if os.path.exists(SIMILARITY_FILE) and not force_recompute:
        similarity_df = pd.read_parquet(SIMILARITY_FILE)
    else:
        grouped_results = []

        for country, country_df in df.groupby("main_country"):
            print(f"Processing {country} ({len(country_df)} companies)...")
            for industry, industry_df in country_df.groupby("main_industry"):
                if len(industry_df) > 1:
                    similarities = calculate_similarities(industry_df)
                    grouped_results.append(similarities)

        similarity_df = pd.concat(grouped_results, ignore_index=True)
        similarity_df.to_parquet(SIMILARITY_FILE, index=False)

    return similarity_df.sort_values(by=["name_similarity", "address_similarity"], ascending=[False, False])
