# preprocessing/imputer.py

import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")

def knn_impute(df, n_neighbors=5):
    """
    Impute missing values using KNN.
    Works for numeric data.
    """
    print(f"Applying KNN Imputer with k={n_neighbors}...")
    imputer = KNNImputer(n_neighbors=n_neighbors)
    imputed_array = imputer.fit_transform(df.select_dtypes(include=[np.number]))
    df[df.select_dtypes(include=[np.number]).columns] = imputed_array
    return df


def missforest_impute(df, max_iter=10, n_estimators=100):
    """
    Iterative imputation using Random Forest.
    Based on the MissForest algorithm from the paper.
    Handles numeric and categorical variables.
    """

    df = df.copy()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    num_cols = df.select_dtypes(include=[np.number]).columns

    # Encode categorical columns
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = df[col].astype(str)  # convert to str to avoid issues
        df[col] = df[col].where(df[col] != 'nan', np.nan)
        notnull = df[col].notnull()
        df.loc[notnull, col] = le.fit_transform(df.loc[notnull, col])
        encoders[col] = le

    prev_df = df.copy()
    for iteration in range(max_iter):
        print(f"MissForest Iteration {iteration + 1}...")
        for col in df.columns:
            if df[col].isnull().sum() == 0:
                continue  # skip fully observed columns

            # Split into observed and missing
            not_null = df[col].notnull()
            X_train = df.loc[not_null].drop(columns=[col])
            y_train = df.loc[not_null, col]
            X_pred = df.loc[~not_null].drop(columns=[col])

            # Choose model based on variable type
            if col in cat_cols:
                model = RandomForestClassifier(n_estimators=n_estimators)
            else:
                model = RandomForestRegressor(n_estimators=n_estimators)

            # Fit and predict
            model.fit(X_train, y_train)
            preds = model.predict(X_pred)
            df.loc[~not_null, col] = preds

        # Check for convergence (simple form)
        if df.equals(prev_df):
            print("Converged.")
            break
        prev_df = df.copy()

    # Decode categorical columns
    for col in cat_cols:
        df[col] = df[col].astype(int)
        df[col] = encoders[col].inverse_transform(df[col])

    return df
