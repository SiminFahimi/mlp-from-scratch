import numpy as np
from typing import Tuple, List
from dataclasses import dataclass

@dataclass
class Dataset:
    X_train: np.ndarray
    Y_train: np.ndarray
    X_test: np.ndarray
    Y_test: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    input_dim: int
 
# SAMPLING / SPLIT
def sample_dataset(X, y, n):
    if n > len(X):
        raise ValueError("n_sample is larger than dataset size")
    idx = np.random.choice(len(X), n, replace=False)
    return X[idx], y[idx]


def global_split(X, y, test_ratio=0.2):
    idx = np.random.permutation(len(X))
    X, y = X[idx], y[idx]

    split = int(len(X) * (1 - test_ratio))
    return X[:split], y[:split], X[split:], y[split:]


 
# FEATURE SELECTION (FIXED)

def select_top_features(X, y, k=2):
    scores = []
    for f in range(X.shape[1]):
        col = X[:, f]
        corr = abs(pearson_correlation(col, y))
        scores.append((f, corr))
    
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:k]


# FEATURE ENGINEERING (GENERIC)
 
def build_features(X, selected_features: List[Tuple[int, float, float]]):
    """Create nonlinear features from selected indices."""
    out = [X]

    for f, _ in selected_features:
        col = X[:, f]
        out.append(col ** 2)
    return np.column_stack(out)

# PREPROCESS PIPELINE
def prep_dataset(X_train_full, y_train_full, X_test, y_test, n_sample, add_feat=False):
    X_train, y_train = sample_dataset(X_train_full, y_train_full, n_sample)

    if add_feat:
        features = select_top_features(X_train, y_train, k=2)
        X_train = build_features(X_train, features)
        X_test = build_features(X_test, features)

    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std == 0] = 1e-8

    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std
    input_dim = X_train.shape[1]
    return Dataset(X_train, y_train, X_test, y_test, mean, std, input_dim)

def pearson_correlation(x, y):
    xm, ym = np.mean(x), np.mean(y)
    num = np.sum((x - xm) * (y - ym))
    den = np.sqrt(np.sum((x - xm) ** 2)) * np.sqrt(np.sum((y - ym) ** 2))
    return num / (den + 1e-8)
