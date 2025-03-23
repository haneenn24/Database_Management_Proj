import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from models.independent_model import evaluate_independent_model
from models.dependent_model import evaluate_dependent_model

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

# Example queries to run (SQL-style)
queries = [
    ("Simple Selection", "SELECT Name, Department FROM df WHERE Salary > 6000"),
    ("Disjoint Projection", "SELECT DISTINCT City FROM df WHERE Age > 30"),
    ("Conjunctive Query", "SELECT * FROM df WHERE Department = 'HR' AND Salary > 5000"),
    ("UCQ Query", """
        SELECT Name FROM df WHERE Department = 'HR'
        UNION
        SELECT Name FROM df WHERE Salary > 6000
    """)
]

results = []

# Run evaluations
for db_name, path in datasets.items():
    try:
        df = pd.read_csv(path)
    except:
        print(f"Skipping {db_name}, file not found.")
        continue

    for query_name, query in queries:
        # Independent model
        start = time.time()
        try:
            result_indep = evaluate_independent_model(df.copy(), query)
        except:
            continue
        time_indep = time.time() - start
        prob_mean_indep = result_indep['final_probability'].mean()

        # Dependent model
        start = time.time()
        try:
            result_dep = evaluate_dependent_model(df.copy(), query)
        except:
            continue
        time_dep = time.time() - start
        prob_mean_dep = result_dep['final_probability'].mean()

        # Save results
        results.append({
            "Dataset": db_name,
            "Query": query_name,
            "Model": "Independent",
            "Runtime": time_indep,
            "Avg_Probability": prob_mean_indep
        })

        results.append({
            "Dataset": db_name,
            "Query": query_name,
            "Model": "Dependent",
            "Runtime": time_dep,
            "Avg_Probability": prob_mean_dep
        })

# Create DataFrame
results_df = pd.DataFrame(results)

# Plot runtimes
plt.figure(figsize=(12, 6))
sns.barplot(data=results_df, x="Dataset", y="Runtime", hue="Model")
plt.title("⏱️ Runtime Comparison Across Models and Datasets")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Plot output probability difference
plt.figure(figsize=(12, 6))
sns.barplot(data=results_df, x="Dataset", y="Avg_Probability", hue="Model")
plt.title("📊 Average Output Probability per Model and Dataset")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()
