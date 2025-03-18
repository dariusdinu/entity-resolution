import pandas as pd


def merge_with_original_data(grouped_df, original_file, output_file):
    original_df = pd.read_parquet(original_file)

    merged_df = original_df.merge(grouped_df[["company_name", "group_id"]],
                                  on="company_name", how="left")

    merged_df.to_parquet(output_file, index=False)
    print(f"Final dataset saved at: {output_file}")

    return merged_df
