import pandas as pd
import networkx as nx
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


def group_similar_companies_rf():
    print("Loading data for Random Forest...\n")
    df = pd.read_parquet("../data/similarity_results.parquet")

    df["label"] = (df["weighted_similarity"] >= 0.75).astype(int)

    print("Label distribution:\n", df["label"].value_counts(), "\n")

    features = df[["name_similarity", "address_similarity", "website_similarity", "weighted_similarity"]]
    labels = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.05, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("Random Forest Evaluation:\n", classification_report(y_test, y_pred), "\n")

    df["predicted"] = model.predict(features)

    matches = df[df["predicted"] == 1]

    G = nx.Graph()
    for _, row in matches.iterrows():
        G.add_edge(row["company_1"], row["company_2"])

    groups = list(nx.connected_components(G))

    group_records = []
    for group_id, group in enumerate(groups, start=1):
        for company in group:
            group_records.append({"company_name": company, "group_id": group_id})

    result_df = pd.DataFrame(group_records)

    # Merge back main_country information
    original_info = df[["company_1", "main_country"]].drop_duplicates().rename(columns={"company_1": "company_name"})
    result_df = result_df.merge(original_info, on="company_name", how="left")

    result_df.to_parquet("../data/grouped_companies_rf.parquet", index=False)
    print("RF-based grouping saved successfully!")

    return result_df
