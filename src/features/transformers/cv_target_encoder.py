from typing import Optional, List, Dict

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import KFold

from src.config import SEED


class CVTargetEncoder(BaseEstimator, TransformerMixin):
    """CV-safe target encoder. Computes OOF encoded values during fit, stores mappings for transform."""

    def __init__(self, n_splits: int = 5, random_state: Optional[int] = None, cols: Optional[List[str]] = None):
        self.n_splits = max(2, int(n_splits))
        self.random_state = SEED if random_state is None else random_state
        self.cols = cols
        self.encoding_map_: Dict[str, Dict] = {}
        self.global_mean_: float = 0.0

    def fit(self, X, y):
        X = pd.DataFrame(X).reset_index(drop=True)
        y = pd.Series(y).reset_index(drop=True)
        if X.shape[0] == 0:
            raise ValueError("CVTargetEncoder received empty X in fit")

        self.global_mean_ = float(y.mean())
        cols = self.cols or X.select_dtypes(include=['object', 'category']).columns.tolist()
        if not cols:
            self.oof_df_ = pd.DataFrame(index=X.index)
            return self

        oof_encoded = {c: np.full(len(X), np.nan, dtype=float) for c in cols}
        n_splits = min(self.n_splits, max(2, X.shape[0]))
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)

        for train_idx, val_idx in kf.split(X):
            y_tr = y.iloc[train_idx]
            X_tr = X.iloc[train_idx]
            X_val = X.iloc[val_idx]
            for c in cols:
                tr_series = X_tr[c].fillna('__NA__')
                val_series = X_val[c].fillna('__NA__')
                tr_means = y_tr.groupby(tr_series).mean()
                oof_vals = val_series.map(tr_means).fillna(self.global_mean_).values
                oof_encoded[c][val_idx] = oof_vals

        for c in cols:
            full_series = X[c].fillna('__NA__')
            full_map = y.groupby(full_series).mean().to_dict()
            self.encoding_map_[c] = full_map

        self.oof_df_ = pd.DataFrame(oof_encoded)
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        for c, mapping in self.encoding_map_.items():
            if c in X.columns:
                mapped = X[c].fillna('__NA__').map(mapping)
                X[c] = mapped.fillna(self.global_mean_)
        return X
