import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from tabulate import tabulate

# File paths
SIMILARITY_FILE = "../data/similarity_results.parquet"
GROUPED_FILE_PARQUET = "../data/grouped_companies.parquet"
GROUPED_FILE_CSV = "../data/grouped_companies.csv"
SIMILARITY_THRESHOLD = 90  # Adjust threshold for stricter clustering


def group_similar_companies():
    df = pd.read_parquet(SIMILARITY_FILE)

    df["name_similarity"] = df["name_similarity"] * 100
    df["address_similarity"] = df["address_similarity"] * 100

    df = df[df["name_similarity"] >= SIMILARITY_THRESHOLD]

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
    grouped_df.head(200).to_csv(GROUPED_FILE_CSV, index=False)  # Only save first 200 for review

    print("\nSample Grouped Companies (First 20 Entries):")
    print(tabulate(grouped_df.head(20), headers="keys", tablefmt="fancy_grid"))

    visualize_group_distribution(grouped_df)

    return grouped_df


def visualize_group_distribution(grouped_df):
    group_sizes = grouped_df["group_id"].value_counts().head(10)

    plt.figure(figsize=(10, 6))
    group_sizes.plot(kind="bar", color="royalblue")
    plt.title("Top 10 Largest Company Groups")
    plt.xlabel("Group ID")
    plt.ylabel("Number of Companies")
    plt.xticks(rotation=45)
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    plt.show()
