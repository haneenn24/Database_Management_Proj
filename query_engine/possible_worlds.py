import pandas as pd
import numpy as np
import itertools

def generate_possible_worlds(df, imputation_dict, tuple_prob_col='Posterior_Prob', n_worlds=5, top_k=2):
    """
    Generate full possible worlds based on:
    1. Probabilistic imputation (top-k values for each missing cell)
    2. Tuple existence probabilities (posterior)
    """

    df_base = df.copy()
    possible_worlds = []

    # ---------------------------
    # STEP 1: Build imputation combinations
    # ---------------------------
    # For each row with missing data, we extract its top-k imputation options
    # Format per row: [(row_idx, value1), (row_idx, value2), ...]
    row_options = []
    row_indices = list(imputation_dict.keys())

    for idx in row_indices:
        # Take top-k imputed values only
        top_k_values = imputation_dict[idx][:top_k]
        # Store (row_index, imputed_value) for each option
        row_options.append([(idx, val[0]) for val in top_k_values])

    # Cartesian product of all possible imputations → all combinations of imputed values
    imputation_combinations = list(itertools.product(*row_options))

    # ---------------------------
    # STEP 2: Construct each possible world
    # ---------------------------
    for i in range(min(n_worlds, len(imputation_combinations))):
        world = df_base.copy()

        # --- Imputation Step ---
        # For each (row, value) pair in this combination,
        # set the missing value in that row to the imputed value
        for row_idx, value in imputation_combinations[i]:
            col_to_impute = df.columns[df.loc[row_idx].isnull()][0]  # only 1 missing per row assumed
            world.at[row_idx, col_to_impute] = value

        # --- Tuple Existence Step ---
        # Sample whether to keep each row based on its Posterior_Prob
        # If Bernoulli trial succeeds, we keep the row in this world
        mask = world[tuple_prob_col].apply(lambda p: np.random.binomial(1, p) == 1)
        final_world = world[mask].reset_index(drop=True)

        possible_worlds.append(final_world)

    return possible_worlds



def generate_possible_worlds(df, imputed_values_dict, tuple_prob_col='Posterior_Prob', n_worlds=5):
    """
    Generate possible worlds by:
    1. Sampling imputed values for missing cells based on their probability distribution.
    2. Sampling tuple inclusion based on posterior probability.

    Parameters:
        df (DataFrame): The original DataFrame with missing values.
        imputed_values_dict (dict): Dictionary of {(row_idx, col_name): [(value, prob), ...]}.
        tuple_prob_col (str): Column name for tuple existence probability.
        n_worlds (int): Number of possible worlds to generate.

    Returns:
        List of DataFrames: One for each possible world.
    """
    possible_worlds = []

    for _ in range(n_worlds):
        df_world = df.copy()

        # Step 1: Impute missing values based on probabilistic top-k values
        for (row_idx, col_name), value_prob_list in imputed_values_dict.items():
            values, probs = zip(*value_prob_list)
            sampled_value = np.random.choice(values, p=probs)
            df_world.at[row_idx, col_name] = sampled_value

        # Step 2: Sample tuple existence
        sampled_indices = []
        for idx, row in df_world.iterrows():
            p_exist = row.get(tuple_prob_col, 1.0)
            if np.random.binomial(1, p_exist):  # keep tuple
                sampled_indices.append(idx)

        df_world = df_world.loc[sampled_indices].reset_index(drop=True)
        possible_worlds.append(df_world)

    return possible_worlds

# Dummy example to simulate execution
example_df = pd.DataFrame({
    'Name': ['Alice', 'Bob', 'Carol'],
    'Department': ['HR', 'IT', 'Finance'],
    'Salary': [6000, np.nan, np.nan],
    'Posterior_Prob': [0.9, 0.6, 0.3]
})

# Simulated top-k imputation results for missing cells
example_imputed_values = {
    (1, 'Salary'): [(5000, 0.7), (5500, 0.3)],
    (2, 'Salary'): [(6500, 0.4), (7000, 0.6)]
}

# Generate 3 example possible worlds
worlds = generate_possible_worlds(example_df, example_imputed_values, 'Posterior_Prob', n_worlds=3)

# Return the first world for visualization
import ace_tools as tools; tools.display_dataframe_to_user(name="Example Possible World 1", dataframe=worlds[0])
