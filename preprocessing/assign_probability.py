# preprocessing/assign_probability.py

import pandas as pd
import numpy as np

def assign_uniform_probabilities(df, low=0.7, high=0.9, seed=42):
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


def assign_bayesian_probabilities(df, evidence_column, event_column, prior_col='Prior_Prob', new_col='Posterior_Prob'):
    """
    Assign or update probabilities of tuple existence using Bayes’ Theorem:
    P(T|E) = [P(E|T) * P(T)] / P(E)

    Parameters:
        df (DataFrame): Input DataFrame.
        evidence_column (str): Column representing evidence E.
        event_column (str): Column representing event T (e.g., class, group).
        prior_col (str): Column name for P(T). If not present, assume uniform prior.
        new_col (str): Output column name for P(T|E) (posterior probability).

    Returns:
        DataFrame with new posterior probability column.
    """
    df = df.copy()

    # If no prior column, assume uniform prior
    if prior_col not in df.columns:
        print("No prior column found. Using uniform prior.")
        df[prior_col] = 1 / df[event_column].nunique()

    # Step 1: Estimate likelihood P(E | T)
    likelihoods = df.groupby(event_column)[evidence_column].mean().to_dict()

    # Step 2: Compute marginal P(E) = ∑ P(E|T) * P(T)
    df['Likelihood'] = df[event_column].map(likelihoods)
    df['Weighted_Likelihood'] = df['Likelihood'] * df[prior_col]
    marginal_e = df['Weighted_Likelihood'].sum()

    # Step 3: Compute posterior: P(T | E) = (P(E|T) * P(T)) / P(E)
    df[new_col] = df['Weighted_Likelihood'] / marginal_e

    # Cleanup intermediate columns
    df.drop(columns=['Likelihood', 'Weighted_Likelihood'], inplace=True)

    return df
