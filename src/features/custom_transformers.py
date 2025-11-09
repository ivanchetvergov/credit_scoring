# src/features/custom_transformers.py
import pandas as pd
import numpy as np
from typing import List
import logging
from sklearn.base import BaseEstimator, TransformerMixin

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AnomalyHandler(BaseEstimator, TransformerMixin):
    """
    Обрабатывает аномальные значения в DAYS_EMPLOYED.
    Заменяет аномалию на NaN и создает бинарный индикатор.
    """

    def __init__(self, anomaly_value=365243):
        self.anomaly_value = anomaly_value
        self.n_anomalies_ = 0

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        if 'DAYS_EMPLOYED' in X.columns:
            # индикатор аномалии (безработный/неизвестно)
            X['DAYS_EMPLOYED_ANOM'] = (X['DAYS_EMPLOYED'] == self.anomaly_value).astype(int)
            self.n_anomalies_ = X['DAYS_EMPLOYED_ANOM'].sum()

            # заменяем аномальные значения на NaN
            X.loc[X['DAYS_EMPLOYED'] == self.anomaly_value, 'DAYS_EMPLOYED'] = np.nan

            logger.debug(f"Anomaly handling: found {X['DAYS_EMPLOYED_ANOM'].sum()} anomalies")
        else:
            logger.warning("Column 'DAYS_EMPLOYED' not found in DataFrame")

        return X


class FeatureCreator(BaseEstimator, TransformerMixin):
    """
    Создает новые, полезные признаки на основе существующих.
    Создаваемые признаки:
    - CREDIT_INCOME_RATIO: отношение кредита к доходу
    - ANNUITY_INCOME_RATIO: отношение аннуитета к доходу
    - AGE_YEARS: возраст клиента в годах
    - EMPLOYMENT_YEARS: стаж работы в годах
    - GOODS_PRICE_TO_CREDIT_RATIO: отношение стоимости товара к кредиту
    """
    def __init__(self, epsilon: float = 1.0):
        self.epsilon = epsilon
        self.created_features_: List[str] = []

    def fit(self, X: pd.DataFrame, y=None) -> 'FeatureCreator':
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        self.created_features_ = []

        # 1 отношение кредита к доходу (credit burden)
        if 'AMT_CREDIT' in X.columns and 'AMT_INCOME_TOTAL' in X.columns:
            X['CREDIT_INCOME_RATIO'] = X['AMT_CREDIT'] / (X['AMT_INCOME_TOTAL'] + 1)
            self.created_features_.append('CREDIT_INCOME_RATIO')

        # 2 отношение аннуитета к доходу (payment burden)
        if 'AMT_ANNUITY' in X.columns and 'AMT_INCOME_TOTAL' in X.columns:
            X['ANNUITY_INCOME_RATIO'] = X['AMT_ANNUITY'] / (X['AMT_INCOME_TOTAL'] + 1)
            self.created_features_.append('ANNUITY_INCOME_RATIO')

        # 3 возраст клиента в годах
        if 'DAYS_BIRTH' in X.columns:
            X['AGE_YEARS'] = (X['DAYS_BIRTH'] / -365).round(1)
            self.created_features_.append('AGE_YEARS')

        # 4 стаж работы в годах
        if 'DAYS_EMPLOYED' in X.columns:
            X['EMPLOYMENT_YEARS'] = (X['DAYS_EMPLOYED'] / -365).clip(lower=0).round(1)
            self.created_features_.append('EMPLOYMENT_YEARS')

        # 5 отношение стоимости товара к кредиту
        if 'AMT_GOODS_PRICE' in X.columns and 'AMT_CREDIT' in X.columns:
            X['GOODS_PRICE_TO_CREDIT_RATIO'] = X['AMT_GOODS_PRICE'] / (X['AMT_CREDIT'] + 1)
            self.created_features_.append('GOODS_PRICE_TO_CREDIT_RATIO')

        logger.debug(f"Feature creation complete. Created {len(self.created_features_)}.")

        return X


class DataFrameCoercer(BaseEstimator, TransformerMixin):
    def __init__(self):
        from src.config import CATEGORICAL_FEATURES, BIN_CATEGORICAL_FEATURES
        self.non_numeric_cols = CATEGORICAL_FEATURES + BIN_CATEGORICAL_FEATURES

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_out = X.copy()
        for col in X_out.columns:
            if col not in self.non_numeric_cols:
                X_out[col] = pd.to_numeric(X_out[col], errors='coerce')
            else:
                # если бинарный признак содержит 'Y'/'N'
                X_out[col] = (
                    X_out[col]
                    .replace({'Y': 1, 'N': 0, 'y': 1, 'n': 0})
                    # .fillna(0)
                )
        return X_out
