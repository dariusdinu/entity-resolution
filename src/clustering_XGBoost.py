import pandas as pd
import xgboost as xgb
import networkx as nx
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# File paths
SIMILARITY_FILE = "../data/similarity_results.parquet"
GROUPED_FILE_PARQUET = "../data/grouped_companies_xgboost.parquet"
GROUPED_FILE_CSV = "../data/grouped_companies_xgboost.csv"

PREDICTION_THRESHOLD = 0.5


def group_similar_companies_ml():
    print("\n📊 Loading data...")
    df = pd.read_parquet(SIMILARITY_FILE)

    # Step 1: Compound pseudo-labeling logic
    df["label"] = (df["weighted_similarity"] >= 0.65).astype(int)

    print("\n🧐 Label distribution:\n", df["label"].value_counts())

    features = df[["name_similarity", "address_similarity", "website_similarity"]]
    labels = df["label"]

    # Step 2: Train-test split
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    # Step 3: Handle imbalance with scale_pos_weight
    pos_weight = (len(y_train) - sum(y_train)) / (sum(y_train) + 1e-6)
    print(f"\n⚖️ Applying scale_pos_weight: {pos_weight:.2f}")

    model = xgb.XGBClassifier(
        use_label_encoder=False,
        eval_metric='logloss',
        base_score=0.1,
        scale_pos_weight=pos_weight
    )
    model.fit(X_train, y_train)

    # Step 4: Evaluate model
    y_pred = model.predict(X_test)
    print("\n🧠 XGBoost Evaluation:\n", classification_report(y_test, y_pred))

    # Step 5: Predict on full dataset
    probabilities = model.predict_proba(features)[:, 1]
    df["ml_prediction"] = (probabilities >= PREDICTION_THRESHOLD).astype(int)

    # Step 6: Build graph from predicted duplicates
    G = nx.Graph()
    for _, row in df[df["ml_prediction"] == 1].iterrows():
        G.add_edge(row["company_1"], row["company_2"])

    # Step 7: Extract clusters
    clusters = list(nx.connected_components(G))
    company_to_group = {}
    for group_id, cluster in enumerate(clusters, start=1):
        for company in cluster:
            company_to_group[company] = group_id

    grouped_df = pd.DataFrame(company_to_group.items(), columns=["company_name", "group_id"])

    # Step 8: Merge additional info like main_country
    additional_cols = df[["company_1", "main_country"]].drop_duplicates().rename(
        columns={"company_1": "company_name"}
    )
    grouped_df = grouped_df.merge(additional_cols, on="company_name", how="left")

    # Step 9: Save results
    grouped_df.to_parquet(GROUPED_FILE_PARQUET, index=False)
    grouped_df.to_csv(GROUPED_FILE_CSV, index=False)
    print("\n✅ ML-based grouping saved successfully!")

    return grouped_df
