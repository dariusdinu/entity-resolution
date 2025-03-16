import pandas as pd
import matplotlib.pyplot as plt

GROUPED_FILE_PARQUET = "../data/grouped_companies.parquet"

def run_insights():
    df = pd.read_parquet(GROUPED_FILE_PARQUET)

    group_sizes = df["group_id"].value_counts()
    print("\nTop 10 Largest Groups:")
    print(group_sizes.head(10))

    sample_group_ids = group_sizes.head(10).index.tolist()
    for group_id in sample_group_ids:
        print(f"\nGroup ID {group_id} Sample:")
        print(df[df["group_id"] == group_id][["company_name", "main_country", "group_id"]].head(5))

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    top_n = 10
    largest_groups = group_sizes.head(top_n)
    largest_groups.plot(kind="bar", color="royalblue", edgecolor="black")
    plt.title(f"Top {top_n} Largest Groups")
    plt.xlabel("Group ID")
    plt.ylabel("Number of Companies")
    plt.grid(True, linestyle="--", alpha=0.6)

    singleton_count = (group_sizes == 1).sum()
    small_groups_count = ((group_sizes > 1) & (group_sizes <= 5)).sum()
    large_groups_count = (group_sizes > 5).sum()

    sizes = [singleton_count, small_groups_count, large_groups_count]
    labels = ['Singletons (1)', 'Small Groups (2-5)', 'Large Groups (6+)']
    colors = ['lightcoral', 'gold', 'lightskyblue']

    plt.subplot(1, 2, 2)
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, colors=colors)
    plt.axis('equal')
    plt.title("Group Size Composition")

    plt.tight_layout()
    plt.show()
