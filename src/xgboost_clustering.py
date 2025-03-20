import networkx as nx
import pandas as pd
import xgboost
from sklearn.model_selection import train_test_split

from constants import XGBOOST_GROUPS_CSV, XGBOOST_SAMPLE_GROUPS_CSV, XGBOOST_GROUPS_PARQUET, \
    XGBOOST_PREDICTION_THRESHOLD, XGBOOST_SIMILARITY_THRESHOLD
from src.insights import run_insights
from src.utils.final_merge import merge_with_original_data


def group_similar_companies_XGBoost(similarity_file, cleaned_file):
    print("Grouping companies using the XGBoost model...")
    df = pd.read_parquet(similarity_file)

    df["label"] = (df["weighted_similarity"] >= XGBOOST_SIMILARITY_THRESHOLD).astype(int)

    features = df[["name_similarity", "address_similarity", "website_similarity"]]
    labels = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    pos_weight = (len(y_train) - sum(y_train)) / (sum(y_train) + 1e-6)

    model = xgboost.XGBClassifier(
        eval_metric='logloss',
        base_score=0.1,
        scale_pos_weight=pos_weight
    )
    model.fit(X_train, y_train)

    probabilities = model.predict_proba(features)[:, 1]
    df["predicted"] = (probabilities >= XGBOOST_PREDICTION_THRESHOLD).astype(int)

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

    grouped_df.to_parquet(XGBOOST_GROUPS_PARQUET, index=False)
    grouped_df.to_csv(XGBOOST_GROUPS_CSV, index=False)
    print("\nXGBoost grouping saved successfully!")

    merge_with_original_data(
        grouped_df,
        original_file=cleaned_file,
        output_file="../data/final_xgboost_dataset.parquet"
    )

    run_insights(XGBOOST_GROUPS_PARQUET, XGBOOST_SAMPLE_GROUPS_CSV)

    return grouped_df
