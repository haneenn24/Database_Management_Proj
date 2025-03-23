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
