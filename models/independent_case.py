# independent_model.py
"""
This module implements the Independent Probabilistic Model.
In this model, all tuples are assumed to be independent events.
To compute the probability of a query result, we multiply the probabilities
of the individual tuples that satisfy the query.

Extensional approach: No dependencies between data values.
"""

import pandas as pd
from functools import reduce


def evaluate_independent_model(results_with_probs):
    """
    Given a list of result tuples and their tuple probabilities,
    compute the final probability of the query result using multiplication.

    Parameters:
    - results_with_probs: List of (tuple, probability) pairs

    Returns:
    - result_table: pd.DataFrame with 'result' and 'probability'
    """
    prob_map = {}

    for row, prob in results_with_probs:
        key = tuple(row)
        if key not in prob_map:
            prob_map[key] = prob
        else:
            # Independent event combination: P1 + P2 - P1*P2
            prev = prob_map[key]
            prob_map[key] = 1 - (1 - prev) * (1 - prob)

    rows = [list(k) + [v] for k, v in prob_map.items()]
    columns = [f"attr_{i}" for i in range(len(rows[0]) - 1)] + ["probability"]
    return pd.DataFrame(rows, columns=columns)

def query_target_probability(results_df, target_row):
    """
    Return the probability of a specific result tuple.

    Parameters:
    - results_df: pd.DataFrame output from evaluate_independent_model
    - target_row: list of attribute values (excluding probability)

    Returns:
    - float: probability value if found, else 0.0
    """
    query = tuple(target_row)
    for _, row in results_df.iterrows():
        if tuple(row[:-1]) == query:
            return row['probability']
    return 0.0

def probability_of_any_match(results_df):
    """
    Compute the probability that at least one result tuple is true (OR logic).

    P(any) = 1 - Π(1 - p_i)
    """
    complement_product = 1.0
    for p in results_df['probability']:
        complement_product *= (1 - p)
    return 1 - complement_product
