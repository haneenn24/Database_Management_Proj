import pandas as pd
import numpy as np
import random

def introduce_missing_data(df, strategy='MCAR', missing_rate=0.2, column='Salary'):
    """
    Introduce missing values in a dataset based on one of the three mechanisms:
    - MCAR: Missing Completely At Random
    - MAR: Missing At Random (depends on another observed attribute)
    - MNAR: Missing Not At Random (depends on the missing value itself)

    Parameters:
        df (DataFrame): Input DataFrame.
        strategy (str): One of 'MCAR', 'MAR', 'MNAR'.
        missing_rate (float): Fraction of values to be made missing.
        column (str): Column to inject missingness into.

    Returns:
        DataFrame with missing values introduced.
    """
    df = df.copy()
    n = len(df)

    if strategy == 'MCAR':
        # Randomly select rows to introduce missing values
        missing_indices = random.sample(range(n), int(missing_rate * n))
        df.loc[missing_indices, column] = np.nan

    elif strategy == 'MAR':
        # Make missingness depend on another attribute (e.g., 'Department')
        if 'Department' not in df.columns:
            raise ValueError("Column 'Department' is required for MAR strategy.")
        mar_condition = df['Department'] == 'IT'
        mar_indices = df[mar_condition].sample(frac=missing_rate).index
        df.loc[mar_indices, column] = np.nan

    elif strategy == 'MNAR':
        # Make missingness depend on the value itself (e.g., values > 5500 more likely to be missing)
        mnar_condition = df[column] > 5500
        mnar_indices = df[mnar_condition].sample(frac=missing_rate).index
        df.loc[mnar_indices, column] = np.nan

    else:
        raise ValueError("Invalid strategy. Choose from 'MCAR', 'MAR', or 'MNAR'.")

    return df


