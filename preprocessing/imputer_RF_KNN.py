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


def knn_impute_deterministic(df, columns, n_neighbors=3):
    """
    Deterministic KNN imputation — fills missing values with a single predicted value.

    Parameters:
        df (DataFrame): Input DataFrame with missing values.
        columns (list): Columns to impute.
        n_neighbors (int): Number of neighbors for KNN.

    Returns:
        DataFrame with missing values imputed.
    """
    imputer = KNNImputer(n_neighbors=n_neighbors)
    df_imputed = df.copy()
    df_imputed[columns] = imputer.fit_transform(df[columns])
    return df_imputed

def knn_impute_probabilistic(df, column, n_neighbors=3, top_k=3):
    """
    Probabilistic KNN imputation — returns top-k candidate values for each missing cell with weights.

    Parameters:
        df (DataFrame): Input DataFrame with missing values.
        column (str): Column to impute probabilistically.
        n_neighbors (int): Number of neighbors for KNN.
        top_k (int): Number of top values to return with probabilities.

    Returns:
        Dict: {index: [(value1, prob1), (value2, prob2), ...]}
    """
    df_clean = df.dropna(subset=[column])
    df_missing = df[df[column].isnull()]
    features = df_clean.drop(columns=[column])
    
    nbrs = NearestNeighbors(n_neighbors=n_neighbors)
    nbrs.fit(features)
    
    prob_output = {}

    for idx in df_missing.index:
        point = df.loc[idx].drop(column).values.reshape(1, -1)
        distances, indices = nbrs.kneighbors(point)
        neighbor_values = df_clean.iloc[indices[0]][column]
        value_counts = neighbor_values.value_counts(normalize=True)
        top_values = value_counts.head(top_k).to_dict()
        prob_output[idx] = list(top_values.items())

    return prob_output


def rf_impute_deterministic(X, max_iter=10, tol=1e-3):
    """
    Impute missing values using Random Forest (deterministic, single value per cell).

    Parameters:
        X (DataFrame): Input DataFrame with missing values.
        max_iter (int): Maximum number of iterations.
        tol (float): Stopping threshold (based on change in imputed values).

    Returns:
        DataFrame: Imputed DataFrame.
    """
    X_imp = X.copy()
    X_old = X.copy()
    n_rows, n_cols = X.shape

    # Step 1: Initial guess for missing values (mean/mode imputation)
    for col in X.columns:
        if X[col].dtype == 'object':
            mode = X[col].mode()[0]
            X_imp[col] = X[col].fillna(mode)
        else:
            mean = X[col].mean()
            X_imp[col] = X[col].fillna(mean)

    # Step 2: Determine order of imputation (least to most missing)
    missing_counts = X.isnull().sum().sort_values()
    sorted_cols = missing_counts[missing_counts > 0].index.tolist()

    # Step 3: Iterate until convergence or max_iter
    for iteration in range(max_iter):
        X_old = X_imp.copy()

        for col in sorted_cols:
            is_categorical = X[col].dtype == 'object'

            # Split observed and missing rows
            obs_mask = X[col].notnull()
            miss_mask = X[col].isnull()

            X_obs = X_imp.loc[obs_mask].drop(columns=[col])
            y_obs = X_imp.loc[obs_mask, col]

            X_miss = X_imp.loc[miss_mask].drop(columns=[col]]

            # Encode categorical target
            if is_categorical:
                le = LabelEncoder()
                y_obs_encoded = le.fit_transform(y_obs.astype(str))
                model = RandomForestClassifier()
            else:
                y_obs_encoded = y_obs
                model = RandomForestRegressor()

            # Train model
            model.fit(X_obs, y_obs_encoded)

            # Predict missing
            y_pred = model.predict(X_miss)

            if is_categorical:
                y_pred = le.inverse_transform(y_pred.astype(int))

            X_imp.loc[miss_mask, col] = y_pred

        # Step 4: Check convergence (mean absolute change)
        diff = (X_imp.select_dtypes(include=[np.number]) - 
                X_old.select_dtypes(include=[np.number])).abs().values.flatten()
        if np.nanmean(diff) < tol:
            break

    return X_imp


def rf_impute_probabilistic(X, column, is_categorical=True, top_k=3):
    """
    Probabilistic imputation using Random Forest:
    For each missing value in the specified column, return top-k predicted values with their probabilities.

    Parameters:
        X (DataFrame): Input DataFrame.
        column (str): Column to impute.
        is_categorical (bool): Whether the column is categorical.
        top_k (int): Number of top predictions to return per missing cell.

    Returns:
        Dict[int, List[Tuple[value, prob]]]: Mapping from index to top-k predictions with probabilities.
    """
    df = X.copy()
    missing_indices = df[df[column].isnull()].index
    df_non_missing = df.dropna(subset=[column])
    df_missing = df.loc[missing_indices].drop(columns=[column])

    y = df_non_missing[column]
    X_features = df_non_missing.drop(columns=[column])

    if is_categorical:
        le = LabelEncoder()
        y_encoded = le.fit_transform(y.astype(str))
        model = RandomForestClassifier(n_estimators=100)
    else:
        y_encoded = y
        model = RandomForestRegressor(n_estimators=100)

    model.fit(X_features, y_encoded)

    prob_outputs = {}

    for idx in missing_indices:
        sample = df_missing.loc[idx].values.reshape(1, -1)

        if is_categorical:
            # Get class probabilities from all trees
            probs = model.predict_proba(sample)[0]
            top_indices = np.argsort(probs)[-top_k:][::-1]
            result = [(le.inverse_transform([i])[0], float(probs[i])) for i in top_indices]
        else:
            # For regression: use predictions from individual trees
            all_preds = np.array([est.predict(sample)[0] for est in model.estimators_])
            unique, counts = np.unique(all_preds.round(2), return_counts=True)
            prob_dist = counts / counts.sum()
            top_indices = np.argsort(prob_dist)[-top_k:][::-1]
            result = [(unique[i], float(prob_dist[i])) for i in top_indices]

        prob_outputs[idx] = result

    return prob_outputs
