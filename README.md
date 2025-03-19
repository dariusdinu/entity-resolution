# Entity Resolution Project

## Overview
The project aims to identify and correctly group duplicate records of companies within a large dataset. The main idea behind its solution is to detect and cluster data entries that are likely to refer to the same entity, giving slight variations in data.

### 1. Data preprocessing 
The first aspect that was taken into consideration was inspecting and refining the dataset. This meant inspecting multiple entries and identifying potential faults. This resulted in a series of rules that had to be applied:
- Filling non-existent `company_name` by using the first value of `company_legal_names`. If that is not available, the first value of `company_commercial_names` is to be used instead.
- In case address information from `main_address_raw_text` is not available, `main_street` is to be used instead.
- Extracting the first value of the `domain` data, where multiple values existed.
- Removed common company suffixes (e.g., "LLC", "Inc")
- Standardized country names with the help of `pycountry` package and `main_country_code`
- Removed special characters 
- Replaced all other missing values `\N` with `None` in order to align with Python standards

### 2. Similarity Calculation
- Calculated **weighted similarity scores** using:
  - `company_name` (70% weight)
  - `address` (20% weight)
  - `domain` (10% weight)
- Applied blocking by **country** to speed up the process.
- Saved similarity results to a `.parquet` file for further clustering.

### 3. Clustering the data
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