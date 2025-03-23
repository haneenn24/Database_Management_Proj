import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# List of dataset paths and their names
datasets = {
    "House Prices": "data/house_prices.csv",
    "Diabetes Health": "data/diabetes_health.csv",
    "Air Quality": "data/air_quality.csv",
    "Adult Census Income": "data/adult_census.csv",
    "Students Performance": "data/students_performance.csv",
    "Financial Distress": "data/financial_distress.csv",
    "Lending Club": "data/lending_club.csv"
}

# Hypothetical mapping of missingness types for illustration
missingness_types = {
    "House Prices": "MCAR",
    "Diabetes Health": "MAR",
    "Air Quality": "MCAR",
    "Adult Census Income": "MNAR",
    "Students Performance": "MAR",
    "Financial Distress": "MAR",
    "Lending Club": "MNAR"
}

# Function to analyze missing values and plot
def analyze_missing_values(file_path, dataset_name):
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    df = pd.read_csv(file_path)
    missing_counts = df.isnull().sum()
    missing_counts = missing_counts[missing_counts > 0]

    if missing_counts.empty:
        print(f"No missing values found in {dataset_name}.")
        return

    print(f"\n📊 Missing values in {dataset_name} ({missingness_types.get(dataset_name, 'Unknown')}):\n{missing_counts}")

    # Plot
    plt.figure(figsize=(10, 6))
    sns.barplot(x=missing_counts.index, y=missing_counts.values, palette="viridis")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Number of Missing Values")
    plt.title(f"{dataset_name} - Missing Values ({missingness_types.get(dataset_name, 'Unknown')})")
    plt.tight_layout()
    plt.show()

# Analyze each dataset
for name, path in datasets.items():
    analyze_missing_values(path, name)

