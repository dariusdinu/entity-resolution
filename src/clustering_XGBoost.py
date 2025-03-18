import pandas as pd
import xgboost as xgb
import networkx as nx
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from insights import run_insights
from utils.final_results_utils import merge_with_original_data

GROUPED_FILE_PARQUET = "../data/grouped_companies_xgboost.parquet"
GROUPED_FILE_CSV = "../data/grouped_companies_xgboost.csv"
SAMPLE_GROUPS_CSV = "../data/sample_groups_rule_xgboost.csv"

PREDICTION_THRESHOLD = 0.5


def group_similar_companies_XGBoost(similarity_file, cleaned_file):
    df = pd.read_parquet(similarity_file)

    df["label"] = (df["weighted_similarity"] >= 0.75).astype(int)

    features = df[["name_similarity", "address_similarity", "website_similarity"]]
    labels = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    pos_weight = (len(y_train) - sum(y_train)) / (sum(y_train) + 1e-6)

    model = xgb.XGBClassifier(
        use_label_encoder=False,
        eval_metric='logloss',
        base_score=0.1,
        scale_pos_weight=pos_weight
    )
    model.fit(X_train, y_train)

    probabilities = model.predict_proba(features)[:, 1]
    df["predicted"] = (probabilities >= PREDICTION_THRESHOLD).astype(int)

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
    print("\nXGBoost grouping saved successfully!")

    merge_with_original_data(
        grouped_df,
        original_file=cleaned_file,
        output_file="../data/final_xgboost_dataset.parquet"
    )

    run_insights(GROUPED_FILE_PARQUET, SAMPLE_GROUPS_CSV)

    return grouped_df
