import networkx as nx
import pandas as pd

# File paths
SIMILARITY_FILE = "../data/similarity_results.parquet"
GROUPED_FILE_PARQUET = "../data/grouped_companies.parquet"
GROUPED_FILE_CSV = "../data/grouped_companies.csv"
SIMILARITY_THRESHOLD = 92  # Adjust threshold for stricter clustering


def group_similar_companies():
    df = pd.read_parquet(SIMILARITY_FILE)

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

    grouped_df.to_parquet(GROUPED_FILE_PARQUET, index=False)
    grouped_df.to_csv(GROUPED_FILE_CSV, index=False)

    return grouped_df
