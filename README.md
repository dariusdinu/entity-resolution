# Entity Resolution Project

## Overview
This project focuses on solving an **Entity Resolution** by identifying and grouping potential duplicate companies within a dataset containing real-world company records. The main objective is to detect and cluster companies that likely refer to the same entity, despite slight variations in the available information.


### 1.Data Preprocessing Pipeline
- Cleaned raw company data from a `.parquet` file.
- Applied a series of data enrichment rules:
  - Filled missing `company_name` by falling back on `company_legal_names` and `company_commercial_names`.
  - Enriched addresses using `main_address_raw_text` and fallback logic.
  - Cleaned domain-related fields (extracting the first value when multiple exist).
- Removed suffixes (e.g., "LLC", "Inc") and standardized key fields for similarity scoring.
- Handled missing values and standardized country names.

### 2. Similarity Calculation
- Calculated **weighted similarity scores** using:
  - `company_name` (70% weight)
  - `address` (20% weight)
  - `domain` (10% weight)
- Applied blocking by **country** to speed up the process.
- Saved similarity results to a `.parquet` file for further clustering.

### 3.Rule-Based Clustering
- Implemented a **graph-based clustering algorithm** using NetworkX to group companies based on a configurable similarity threshold.
- Successfully generated clusters (group IDs) and saved the grouped output.

### 4. Insights & Visualizations
- Developed a reporting module to:
  - Display and export **sample clusters** (top 10 largest clusters).
  - Plot:
    - A **bar chart** for the largest clusters.
    - A **pie chart** showing overall group size distribution (e.g., % of singletons vs. multi-company clusters).
  - Automatically export sample groups to a CSV file.

## How to Run the Pipeline

### 1. **Preprocessing (optional if already done):**
```python
# Inside main.py
# df = preprocess_data()
```

### 2. **Similarity Calculation (optional if already done):**
```python
# Inside main.py
# final_similarity_df = find_similar_companies(df)
```

### 3. **Clustering**
```python
# group_similar_companies()
```

### 4. **Insights & Exports:**
```python
# run_insights()
```