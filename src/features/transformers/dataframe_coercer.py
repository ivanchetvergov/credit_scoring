from typing import Optional, List
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from configs.raw_features_config import BIN_CATEGORICAL_FEATURES

class DataFrameCoercer(BaseEstimator, TransformerMixin):
    """
    Приводит типы данных к нужному формату перед обучением.
    - Числовые колонки -> float/int
    - Бинарные (Y/N) -> 0/1
    - Категориальные -> оставляет как есть
    """

    def __init__(self):
        self.non_numeric_cols: Optional[List[str]] = None

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        # определяем нечисловые колонки для защиты от коэрции
        self.non_numeric_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        # расширяем за счет сконфигурированных бинарных категориальных фич (защитно)
        self.non_numeric_cols = list(set(self.non_numeric_cols + list(BIN_CATEGORICAL_FEATURES)))
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        X_out = X.copy()

        # перебираем все колонки и коэрцируем те, которые не в списке
        for col in X_out.columns:
            if col not in self.non_numeric_cols:
                # попытка преобразовать в число, 'coerce' заменяет ошибки на NaN
                X_out[col] = pd.to_numeric(X_out[col], errors='coerce')
            elif col in X_out.columns:
                X_out[col] = X_out[col].replace({'Y': 1, 'N': 0, 'y': 1, 'n': 0})
        return X_out
