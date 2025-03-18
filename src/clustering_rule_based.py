import networkx as nx
import pandas as pd

from insights import run_insights
from utils.final_results_utils import merge_with_original_data

GROUPED_FILE_PARQUET = "../data/grouped_companies_rule_based.parquet"
GROUPED_FILE_CSV = "../data/grouped_companies_rule_based.csv"
SAMPLE_GROUPS_CSV = "../data/sample_groups_rule_based.csv"

SIMILARITY_THRESHOLD = 92


def group_similar_companies(similarity_file, cleaned_file):
    df = pd.read_parquet(similarity_file)

    df["weighted_similarity"] = df["weighted_similarity"] * 100
    df = df[df["weighted_similarity"] >= SIMILARITY_THRESHOLD]

    G = nx.Graph()
    for _, row in df.iterrows():
        G.add_edge(row["company_1"], row["company_2"])

    clusters = list(nx.connected_components(G))

    company_to_group = {}
    for group_id, cluster in enumerate(clusters, start=1):
        for company in cluster:
            company_to_group[company] = group_id

    grouped_df = pd.DataFrame(company_to_group.items(), columns=["company_name", "group_id"])

    additional_cols = df[["company_1", "main_country"]].drop_duplicates().rename(
        columns={"company_1": "company_name"}
    )

    grouped_df = grouped_df.merge(additional_cols, on="company_name", how="left")

    grouped_df.to_parquet(GROUPED_FILE_PARQUET, index=False)
    grouped_df.to_csv(GROUPED_FILE_CSV, index=False)

    print("Grouped companies with country info saved successfully!")

    merge_with_original_data(
        grouped_df,
        original_file=cleaned_file,
        output_file="../data/final_rule_based_dataset.parquet"
    )

    run_insights(GROUPED_FILE_PARQUET, SAMPLE_GROUPS_CSV)

    return grouped_df
