# src/custom_transformers.py

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class RemoveCollinearFeatures(BaseEstimator, TransformerMixin):
    """Custom transformer to remove highly collinear features based on a correlation threshold."""

    def __init__(self, threshold=0.9):
        self.threshold = threshold
        self.to_drop = None

    def fit(self, X, y=None):
        corr_matrix = pd.DataFrame(X).corr().abs()
        upper_triangle = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        self.to_drop = [
            column
            for column in upper_triangle.columns
            if any(upper_triangle[column] > self.threshold)
        ]
        return self

    def transform(self, X):
        return pd.DataFrame(X).drop(self.to_drop, axis=1, errors="ignore").values
