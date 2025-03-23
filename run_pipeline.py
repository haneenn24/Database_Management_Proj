import duckdb
import pandas as pd
from pgmpy.models import BayesianModel
from pgmpy.inference import VariableElimination


def evaluate_independent_model(df, sql_query):
    """
    Evaluate the SQL query using DuckDB assuming independent tuple probabilities.
    """
    con = duckdb.connect()
    con.register("df", df)
    result = con.execute(sql_query).df()

    if 'probability' in result.columns:
        # Combine probabilities if multiple tuples match (simulate product if needed)
        result['final_probability'] = result['probability']
    else:
        result['final_probability'] = 1.0

    return result


def evaluate_dependent_model(df, sql_query):
    """
    Evaluate query assuming tuple dependencies using Bayesian Network (simplified demo).
    This is a placeholder for real conditional inference.
    """
    con = duckdb.connect()
    con.register("df", df)
    result = con.execute(sql_query).df()

    # Simulate dependency-aware probability using placeholder value
    result['final_probability'] = 0.5  # Would be derived using Bayesian inference
    return result


def run_query_ui():
    print("🎯 Welcome to the Probabilistic Query Interface\n")

    # Model selection
    model = input("Select model [independent / dependent]: ").strip().lower()

    # Load dataset
    csv_path = input("Enter CSV path (with a 'probability' column): ").strip()
    try:
        df = pd.read_csv(csv_path)
    except:
        print("❌ Failed to read CSV.")
        return

    # Show example queries
    print("\n🧠 Example Queries You Can Try:")
    print("1. SELECT Name, Department FROM df WHERE Salary > 6000")
    print("2. SELECT DISTINCT City FROM df WHERE Age > 30")
    print("3. SELECT * FROM df1 JOIN df2 ON df1.ID = df2.ID")
    print("4. SELECT Name FROM df WHERE Department = 'HR'\n   UNION\n   SELECT Name FROM df WHERE Salary > 6000\n")

    # Query input
    print("Type your SQL query below:")
    sql_query = input(">>> ")

    # Evaluate
    if model == "independent":
        result = evaluate_independent_model(df, sql_query)
    elif model == "dependent":
        result = evaluate_dependent_model(df, sql_query)
    else:
        print("❌ Invalid model type.")
        return

    print("\n✅ Query Result:")
    print(result.head(10))


if __name__ == "__main__":
    run_query_ui()


#Inputs we can run:
-- Selection + projection
SELECT Name, Department FROM df WHERE Salary > 6000

-- Disjoint projection
SELECT DISTINCT City FROM df WHERE Age > 30

-- Conjunctive query
SELECT * FROM df WHERE Department = 'HR' AND Salary > 5000

-- UCQ (Union of conjunctive queries)
SELECT Name FROM df WHERE Department = 'HR'
UNION
SELECT Name FROM df WHERE Salary > 6000
