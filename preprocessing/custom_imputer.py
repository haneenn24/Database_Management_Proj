
"""
Custom Imputation Methods for Final Project (No Built-in Scikit Imputers)
"""

import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.preprocessing import LabelEncoder
from scipy.spatial.distance import cdist


def knn_impute_custom(df, column, n_neighbors=3):
    """
    KNN-based imputation without sklearn KNNImputer.
    Computes distances manually.
    """
    df = df.copy()
    df_numeric = df.select_dtypes(include=[np.number])
    if column not in df_numeric.columns:
        raise ValueError(f"Column {column} must be numeric")

    target_missing = df[df[column].isnull()]
    target_known = df[df[column].notnull()]
    features = [col for col in df_numeric.columns if col != column]

    for idx, row in target_missing.iterrows():
        # Compute distances to known values
        known_vals = target_known[features]
        target_vals = row[features].values.reshape(1, -1)
        distances = cdist(known_vals, target_vals)[..., 0]
        nearest_idx = distances.argsort()[:n_neighbors]
        nearest_vals = target_known.iloc[nearest_idx][column]
        imputed_value = nearest_vals.mean()
        df.loc[idx, column] = imputed_value

    return df


def rf_impute_custom(df, column, is_categorical=False, n_trees=5):
    """
    Custom Random Forest-style imputation using multiple decision trees.
    No sklearn ensemble used — simple illustrative logic.
    """
    df = df.copy()
    df_full = df.dropna(subset=[column])
    df_missing = df[df[column].isnull()]
    features = [col for col in df.columns if col != column]

    # Encode categorical
    le = None
    if is_categorical:
        le = LabelEncoder()
        df_full[column] = le.fit_transform(df_full[column].astype(str))

    predictions = defaultdict(list)

    for i in range(n_trees):
        sample = df_full.sample(frac=0.8, replace=True)
        X_train = sample[features]
        y_train = sample[column]

        tree = DecisionTreeClassifier() if is_categorical else DecisionTreeRegressor()
        tree.fit(X_train, y_train)

        for idx, row in df_missing.iterrows():
            pred = tree.predict([row[features]])[0]
            predictions[idx].append(pred)

    for idx, preds in predictions.items():
        if is_categorical:
            mode = Counter(preds).most_common(1)[0][0]
            df.loc[idx, column] = le.inverse_transform([int(mode)])[0]
        else:
            df.loc[idx, column] = np.mean(preds)

    return df
