"""
This function implements possible worlds semantics.

Each tuple in the dataset may have:
1. Missing attribute values (e.g., Carol's salary = 6000 (0.5), 7000 (0.5))
2. Uncertain tuple existence (e.g., Bob's tuple has 0.8 probability to exist)

We do:
- For each uncertain tuple or attribute:
  - Enumerate all possible options with their probabilities
- Build all combinations (Cartesian product of options)
- Each combination = one possible world
- Output: a list of DataFrames (possible worlds) with associated world-level probabilities

This simulates the full semantics of a probabilistic database.

In Summary:
This function:

Simulates multiple valid interpretations (worlds) of your uncertain database

Generates each possible world by resolving uncertain values

Assigns a total probability to each world using Bayes-like product rules
"""
import pandas as pd
import itertools

def generate_possible_worlds(df, uncertain_columns_info):
    """
    Implements the Possible Worlds semantics for probabilistic databases.

    A possible world is a fully resolved version of the database:
    - All missing values are filled with one of their possible options.
    - Each uncertain value or tuple has an associated probability.
    - The total probability of a world is the product of all chosen values' probabilities.

    Parameters:
    ----------
    df : pd.DataFrame
        The original dataframe with some missing or uncertain values.

    uncertain_columns_info : dict
        Dictionary mapping row index to a dictionary of uncertain columns.
        Each inner dict maps column name to a list of (value, probability) tuples.

        Example:
        {
            0: {'Department': [('HR', 0.6), ('IT', 0.4)]},
            1: {'Salary': [(6000, 0.5), (7000, 0.5)]}
        }

    Returns:
    -------
    possible_worlds : list of tuples
        Each item is (world_df, world_probability)
        where world_df is a resolved DataFrame and world_probability is a float.
    """

    uncertain_rows = []

    for row_idx, cols in uncertain_columns_info.items():
        options = []
        for col, values_probs in cols.items():
            options.append([(col, val, prob) for val, prob in values_probs])
        uncertain_rows.append((row_idx, list(itertools.product(*options))))

    all_combinations = list(itertools.product(*[r[1] for r in uncertain_rows]))

    possible_worlds = []
    for combo in all_combinations:
        temp_df = df.copy()
        total_prob = 1.0

        for i, row_values in enumerate(combo):
            row_idx = uncertain_rows[i][0]
            for col, val, prob in row_values:
                temp_df.at[row_idx, col] = val
                total_prob *= prob

        possible_worlds.append((temp_df.copy(), total_prob))

    return possible_worlds


#Example Usage:
df = pd.DataFrame({
    "Name": ["Bob", "Carol"],
    "Department": [None, "IT"],
    "Salary": [8000, None]
})

uncertain_columns_info = {
    0: {'Department': [('HR', 0.6), ('IT', 0.4)]},
    1: {'Salary': [(6000, 0.5), (7000, 0.5)]}
}

from utils.possible_worlds import generate_possible_worlds
worlds = generate_possible_worlds(df, uncertain_columns_info)

# View each possible world and its probability
for i, (world, prob) in enumerate(worlds):
    print(f"\n🌍 World {i+1} (P = {prob:.3f})")
    print(world)
