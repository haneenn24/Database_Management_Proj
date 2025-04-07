
import numpy as np
from collections import Counter, defaultdict
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.preprocessing import LabelEncoder
from scipy.spatial.distance import cdist


def knn_impute_topk(df, column, n_neighbors=5, top_k=3):
    """
    Custom probabilistic KNN imputation: top-k values and their frequencies.

    Parameters:
        df (DataFrame): Input data
        column (str): Column to impute
        n_neighbors (int): Number of neighbors to consider
        top_k (int): Number of top candidate values to return

    Returns:
        Dict: {index: [(val1, prob1), (val2, prob2), ...]}
    """
    df = df.copy()
    df_numeric = df.select_dtypes(include=[np.number])
    if column not in df_numeric.columns:
        raise ValueError(f"Column {column} must be numeric")

    target_missing = df[df[column].isnull()]
    target_known = df[df[column].notnull()]
    features = [col for col in df_numeric.columns if col != column]

    result = {}
    for idx, row in target_missing.iterrows():
        known_vals = target_known[features]
        target_vals = row[features].values.reshape(1, -1)
        distances = cdist(known_vals, target_vals)[..., 0]
        nearest_idx = distances.argsort()[:n_neighbors]
        neighbor_vals = target_known.iloc[nearest_idx][column]

        value_counts = neighbor_vals.round(2).value_counts(normalize=True).head(top_k)
        result[idx] = list(value_counts.items())

    return result


def rf_impute_topk(df, column, is_categorical=True, n_trees=5, top_k=3):
    """
    Custom Random Forest-style probabilistic imputation using multiple decision trees.

    Parameters:
        df (DataFrame): Input DataFrame
        column (str): Column to impute
        is_categorical (bool): Whether the column is categorical
        n_trees (int): Number of trees
        top_k (int): Number of top values to return with probabilities

    Returns:
        Dict[int, List[Tuple[value, prob]]]: Mapping from row index to top-k value-prob pairs
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

    result = {}
    for idx, preds in predictions.items():
        if is_categorical:
            counts = Counter(preds)
            most_common = counts.most_common(top_k)
            if le:
                most_common = [(le.inverse_transform([int(k)])[0], v/len(preds)) for k, v in most_common]
        else:
            unique, counts = np.unique(np.round(preds, 2), return_counts=True)
            probs = counts / counts.sum()
            top_idx = np.argsort(probs)[-top_k:][::-1]
            most_common = [(unique[i], float(probs[i])) for i in top_idx]
        result[idx] = most_common

    return result
