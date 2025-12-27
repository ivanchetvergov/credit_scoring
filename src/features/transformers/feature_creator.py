import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from . import logger


class FeatureCreator(BaseEstimator, TransformerMixin):
    """Создает производные признаки из raw data: temporal, ratios, domain-specific."""

    def __init__(self, anomaly_value=365243, epsilon=1e-5):
        self.anomaly_value = anomaly_value
        self.epsilon = epsilon
        self.created_features_ = []

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        X = X.copy()
        self.created_features_ = []

        # Temporal features
        if 'DAYS_BIRTH' in X.columns:
            X['AGE_YEARS'] = (-X['DAYS_BIRTH'] / 365).round(1)
            X['AGE_GROUP'] = pd.cut(X['AGE_YEARS'], bins=[0, 25, 35, 45, 55, 100], labels=[1, 2, 3, 4, 5])
            self.created_features_.extend(['AGE_YEARS', 'AGE_GROUP'])

        if 'DAYS_EMPLOYED' in X.columns:
            X['DAYS_EMPLOYED_ANOM'] = (X['DAYS_EMPLOYED'] == self.anomaly_value).astype(int)
            X.loc[X['DAYS_EMPLOYED'] == self.anomaly_value, 'DAYS_EMPLOYED'] = np.nan
            X['EMPLOYMENT_YEARS'] = (-X['DAYS_EMPLOYED'] / 365).clip(lower=0).round(1)
            self.created_features_.extend(['DAYS_EMPLOYED_ANOM', 'EMPLOYMENT_YEARS'])

        if 'DAYS_REGISTRATION' in X.columns:
            X['REGISTRATION_YEARS'] = (-X['DAYS_REGISTRATION'] / 365).round(1)
            self.created_features_.append('REGISTRATION_YEARS')

        if 'DAYS_ID_PUBLISH' in X.columns:
            X['ID_PUBLISH_YEARS'] = (-X['DAYS_ID_PUBLISH'] / 365).round(1)
            self.created_features_.append('ID_PUBLISH_YEARS')

        # Credit burden ratios
        if 'AMT_CREDIT' in X.columns and 'AMT_INCOME_TOTAL' in X.columns:
            X['CREDIT_INCOME_RATIO'] = X['AMT_CREDIT'] / (X['AMT_INCOME_TOTAL'] + self.epsilon)
            X['CREDIT_BURDEN_GROUP'] = pd.cut(X['CREDIT_INCOME_RATIO'], bins=[0, 5, 10, 50], labels=['low', 'medium', 'high'])
            self.created_features_.extend(['CREDIT_INCOME_RATIO', 'CREDIT_BURDEN_GROUP'])

        if 'AMT_ANNUITY' in X.columns and 'AMT_INCOME_TOTAL' in X.columns:
            X['ANNUITY_INCOME_RATIO'] = X['AMT_ANNUITY'] / (X['AMT_INCOME_TOTAL'] + self.epsilon)
            self.created_features_.append('ANNUITY_INCOME_RATIO')

        if 'AMT_ANNUITY' in X.columns and 'AMT_CREDIT' in X.columns:
            X['ANNUITY_CREDIT_RATIO'] = X['AMT_ANNUITY'] / (X['AMT_CREDIT'] + self.epsilon)
            X['CREDIT_TERM_EST'] = X['AMT_CREDIT'] / (X['AMT_ANNUITY'] + self.epsilon)
            self.created_features_.extend(['ANNUITY_CREDIT_RATIO', 'CREDIT_TERM_EST'])

        if 'AMT_GOODS_PRICE' in X.columns and 'AMT_CREDIT' in X.columns:
            X['GOODS_PRICE_CREDIT_RATIO'] = X['AMT_GOODS_PRICE'] / (X['AMT_CREDIT'] + self.epsilon)
            X['DOWN_PAYMENT_PROXY'] = (X['AMT_CREDIT'] - X['AMT_GOODS_PRICE']).clip(lower=0)
            self.created_features_.extend(['GOODS_PRICE_CREDIT_RATIO', 'DOWN_PAYMENT_PROXY'])

        # Income per capita
        if 'AMT_INCOME_TOTAL' in X.columns and 'CNT_FAM_MEMBERS' in X.columns:
            X['INCOME_PER_PERSON'] = X['AMT_INCOME_TOTAL'] / (X['CNT_FAM_MEMBERS'] + self.epsilon)
            self.created_features_.append('INCOME_PER_PERSON')

        # External source combinations
        ext_cols = [c for c in ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3'] if c in X.columns]
        if len(ext_cols) >= 2:
            X['EXT_SOURCE_MEAN'] = X[ext_cols].mean(axis=1)
            X['EXT_SOURCE_MIN'] = X[ext_cols].min(axis=1)
            X['EXT_SOURCE_MAX'] = X[ext_cols].max(axis=1)
            X['EXT_SOURCE_RANGE'] = X['EXT_SOURCE_MAX'] - X['EXT_SOURCE_MIN']
            self.created_features_.extend(['EXT_SOURCE_MEAN', 'EXT_SOURCE_MIN', 'EXT_SOURCE_MAX', 'EXT_SOURCE_RANGE'])

        # Document count
        doc_cols = [c for c in X.columns if c.startswith('FLAG_DOCUMENT_')]
        if doc_cols:
            X['DOCUMENTS_SUBMITTED'] = X[doc_cols].sum(axis=1)
            self.created_features_.append('DOCUMENTS_SUBMITTED')

        # Interaction squares
        for col in ['CREDIT_INCOME_RATIO', 'ANNUITY_INCOME_RATIO', 'AGE_YEARS', 'EMPLOYMENT_YEARS']:
            if col in X.columns:
                X[f'{col}_SQ'] = X[col] ** 2
                self.created_features_.append(f'{col}_SQ')

        logger.debug(f"FeatureCreator: created {len(self.created_features_)} features")
        return X
