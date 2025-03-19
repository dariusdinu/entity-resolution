import networkx as nx
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from src.insights import run_insights
from src.utils.final_merge import merge_with_original_data

GROUPED_FILE_PARQUET = "../data/grouped_companies_rf.parquet"
GROUPED_FILE_CSV = "../data/grouped_companies_rf.csv"
SAMPLE_GROUPS_CSV = "../data/sample_groups_rule_rf.csv"


def group_similar_companies_rf(similarity_file, cleaned_file):
    df = pd.read_parquet(similarity_file)

    df["label"] = (df["weighted_similarity"] >= 0.75).astype(int)

    features = df[["name_similarity", "address_similarity", "website_similarity", "weighted_similarity"]]
    labels = df["label"]

    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
    model.fit(features, labels)

    df["predicted"] = model.predict(features)

    G = nx.Graph()
    for _, row in df[df["predicted"] == 1].iterrows():
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
    print("\nRandom Forest grouping saved successfully!")

    merge_with_original_data(
        grouped_df,
        original_file=cleaned_file,
        output_file="../data/final_rf_dataset.parquet"
    )

    run_insights(GROUPED_FILE_PARQUET, SAMPLE_GROUPS_CSV)

    return grouped_df
