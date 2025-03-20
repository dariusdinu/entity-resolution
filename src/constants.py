# File paths
INITIAL_FILE = "../data/raw_companies.parquet"
CLEANED_FILE = "../data/cleaned_companies.parquet"
SIMILARITY_FILE = "../data/similarity_results.parquet"

# Output files
RULE_BASED_GROUPS_PARQUET = "../data/grouped_companies_rule_based.parquet"
RULE_BASED_GROUPS_CSV = "../data/grouped_companies_rule_based.csv"
RULE_BASED_SAMPLE_GROUPS_CSV = "../data/sample_groups_rule_based.csv"

RF_GROUPS_PARQUET = "../data/grouped_companies_rf.parquet"
RF_GROUPS_CSV = "../data/grouped_companies_rf.csv"
RF_SAMPLE_GROUPS_CSV = "../data/sample_groups_rule_rf.csv"

XGBOOST_GROUPS_PARQUET = "../data/grouped_companies_xgboost.parquet"
XGBOOST_GROUPS_CSV = "../data/grouped_companies_xgboost.csv"
XGBOOST_SAMPLE_GROUPS_CSV = "../data/sample_groups_rule_xgboost.csv"

# Thresholds
RULE_BASED_SIMILARITY_THRESHOLD = 92
RF_SIMILARITY_THRESHOLD = 0.75
XGBOOST_SIMILARITY_THRESHOLD = 0.75
XGBOOST_PREDICTION_THRESHOLD = 0.5
