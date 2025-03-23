import pandas as pd

def run_query_on_world(df, query):
    """
    Execute a supported SQL-style query on a single possible world.
    Returns a DataFrame of results for that world (no prob column yet).
    """
    # Support: selection, projection, join, union of conjunctive queries
    # For now we simulate this with DataFrame filters
    try:
        result = df.query(query["where"]) if query.get("where") else df
        if query.get("select"):
            result = result[query["select"]]
        if query.get("distinct"):
            result = result.drop_duplicates()
        return result
    except Exception as e:
        print(f"Query failed on world: {e}")
        return pd.DataFrame()


def evaluate_query_on_possible_worlds(worlds, query):
    """
    Apply query to all possible worlds and aggregate the probabilistic result set.
    Each tuple gets a probability = sum over worlds where it appears, weighted by world prob.
    """
    result_prob_map = {}

    for df, world_prob in worlds:
        result = run_query_on_world(df, query)

        for _, row in result.iterrows():
            row_tuple = tuple(row)
            result_prob_map[row_tuple] = result_prob_map.get(row_tuple, 0.0) + world_prob

    # Create final result table
    result_rows = [list(row) + [prob] for row, prob in result_prob_map.items()]
    columns = query["select"] + ["probability"]
    return pd.DataFrame(result_rows, columns=columns)



#Usage Example:
Example Query (Conjunctive)

query = {
    "select": ["Name", "Department"],
    "where": "Salary > 6000 and Department == 'HR'",
    "distinct": True
}

from query_engine import evaluate_query_on_possible_worlds

results = evaluate_query_on_possible_worlds(worlds, query)
print(results)

🔁 Support for UCQ (Union of Conjunctive Queries)
Add support like:

python
Copy
Edit
ucq_query = [
    {"select": ["Name"], "where": "Salary > 6000"},
    {"select": ["Name"], "where": "Department == 'HR'"}
]

def evaluate_ucq(worlds, ucq):
    all_results = pd.DataFrame()
    for subquery in ucq:
        partial = evaluate_query_on_possible_worlds(worlds, subquery)
        all_results = pd.concat([all_results, partial])
    return all_results.groupby(list(all_results.columns[:-1])).agg({"probability": "sum"}).reset_index()