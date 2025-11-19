import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from . import logger

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
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
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
