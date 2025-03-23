# preprocessing/assign_probability.py

import pandas as pd
import numpy as np

def initialize_prior_probabilities(df, low=0.7, high=0.9, seed=42):
    """
    Randomly assign a prior probability to each row from Uniform(low, high).
    """
    np.random.seed(seed)
    priors = np.random.uniform(low, high, size=len(df))
    df['prior_probability'] = priors
    return df

def bayesian_update_probabilities(df, likelihood_missing=0.8):
    """
    Update probabilities using Bayes' Rule approximation for missing data.
    Assumes P(E) is constant across rows.
    """
    if 'prior_probability' not in df.columns:
        raise ValueError("Prior probabilities must be assigned first.")

    df = df.copy()
    total_columns = df.shape[1] - 1  # Exclude 'prior_probability'

    def update_row_probability(row):
        num_missing = row.isnull().sum()
        # Use simplified Bayes' rule: P(T|E) ≈ P(E|T) * P(T)
        P_T = row['prior_probability']
        # Assume evidence likelihood drops with more missing fields
        if total_columns == 0:
            return P_T
        P_E_given_T = (1 - num_missing / total_columns) * likelihood_missing
        return P_T * P_E_given_T

    df['probability'] = df.apply(update_row_probability, axis=1)
    return df
