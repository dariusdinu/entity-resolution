import networkx as nx
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from src.insights import run_insights
from src.utils.final_merge import merge_with_original_data
from constants import RF_SIMILARITY_THRESHOLD, RF_GROUPS_CSV, RF_GROUPS_PARQUET, RF_SAMPLE_GROUPS_CSV


def group_similar_companies_rf(similarity_file, cleaned_file):
    print("Grouping companies using the Random Forest model...")

    df = pd.read_parquet(similarity_file)

    df["label"] = (df["weighted_similarity"] >= RF_SIMILARITY_THRESHOLD).astype(int)

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

    grouped_df.to_parquet(RF_GROUPS_PARQUET, index=False)
    grouped_df.to_csv(RF_GROUPS_CSV, index=False)
    print("\nRandom Forest grouping saved successfully!")

    merge_with_original_data(
        grouped_df,
        original_file=cleaned_file,
        output_file="../data/final_rf_dataset.parquet"
    )

    run_insights(RF_GROUPS_PARQUET, RF_SAMPLE_GROUPS_CSV)

    return grouped_df
