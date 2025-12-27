from typing import Optional, Dict

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from . import logger


class AuxiliaryFeatureAggregator(BaseEstimator, TransformerMixin):
    """Агрегирует данные из вспомогательных таблиц на уровень клиента (SK_ID_CURR)."""

    def __init__(self, aux_data: Optional[Dict[str, pd.DataFrame]] = None):
        self._fitted_stats = None
        self.aux_data = aux_data or {}
        self.bureau_agg_ = None
        self.prev_agg_ = None
        self.agg_funcs = ['mean', 'max', 'min', 'sum', 'std', self._q25, self._q75]

    @staticmethod
    def _q25(x):
        return x.quantile(0.25)

    @staticmethod
    def _q75(x):
        return x.quantile(0.75)

    def fit(self, X: pd.DataFrame, y=None):
        self._fitted_stats = []
        logger.info("Fitting AuxiliaryFeatureAggregator...")

        df_bb = self.aux_data.get('bureau_balance', pd.DataFrame())
        df_bb_agg = self._aggregate_bureau_balance(df_bb)

        df_bureau = self.aux_data.get('bureau', pd.DataFrame())
        self.bureau_agg_ = self._aggregate_bureau(df_bureau, df_bb_agg)

        df_prev = self.aux_data.get('previous_application', pd.DataFrame())
        self.prev_agg_ = self._aggregate_previous_application(df_prev)

        logger.info("AuxiliaryFeatureAggregator fitted successfully")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Применяет предвычисленные агрегации через merge."""
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        X_out = X.copy()

        if self.bureau_agg_ is not None and not self.bureau_agg_.empty:
            X_out = X_out.merge(self.bureau_agg_, on='SK_ID_CURR', how='left')

        if self.prev_agg_ is not None and not self.prev_agg_.empty:
            X_out = X_out.merge(self.prev_agg_, on='SK_ID_CURR', how='left')

        return X_out

    @staticmethod
    def _aggregate_bureau_balance(df_bb: pd.DataFrame) -> pd.DataFrame:
        """Агрегирует статусы просрочки из bureau_balance на уровень SK_ID_BUREAU."""
        if df_bb.empty:
            return pd.DataFrame()

        df_bb_agg = (
            df_bb
            .groupby('SK_ID_BUREAU')['STATUS']
            .value_counts(normalize=True)
            .unstack(fill_value=0)
        )
        df_bb_agg.columns = [f'BB_STATUS_PCT_{col}' for col in df_bb_agg.columns]
        df_bb_agg['BB_HISTORY_LENGTH'] = df_bb.groupby('SK_ID_BUREAU').size()

        return df_bb_agg.reset_index()

    def _aggregate_bureau(self, df_bureau: pd.DataFrame, df_bb_agg: pd.DataFrame) -> pd.DataFrame:
        """Агрегирует данные Bureau на уровень клиента."""
        if df_bureau.empty:
            return pd.DataFrame()

        if not df_bb_agg.empty:
            df_bureau = df_bureau.merge(df_bb_agg, on='SK_ID_BUREAU', how='left')

        num_cols = [
            c for c in df_bureau.select_dtypes(include=[np.number]).columns
            if c not in ['SK_ID_CURR', 'SK_ID_BUREAU']
        ]

        agg_dict = {col: self.agg_funcs for col in num_cols}
        agg_dict['SK_ID_BUREAU'] = ['count']

        df_agg = df_bureau.groupby('SK_ID_CURR').agg(agg_dict)

        new_cols = []
        for col in df_agg.columns:
            col_name, func = col[0], col[1]
            if col_name == 'SK_ID_BUREAU':
                new_cols.append('BUREAU_CREDIT_COUNT')
                continue

            if isinstance(func, str):
                func_label = func.upper()
            elif callable(func):
                fname = getattr(func, '__name__', '')
                if fname in ('_q25', 'q25'):
                    func_label = 'P25'
                elif fname in ('_q75', 'q75'):
                    func_label = 'P75'
                else:
                    func_label = fname.upper() if fname else 'FUNC'
            else:
                func_label = str(func).upper()

            new_cols.append(f'BUREAU_{col_name}_{func_label}')

        df_agg.columns = new_cols
        return df_agg.reset_index()

    @staticmethod
    def _aggregate_previous_application(df_prev: pd.DataFrame) -> pd.DataFrame:
        """Агрегирует данные о предыдущих заявках на уровень клиента."""
        if df_prev.empty:
            return pd.DataFrame()

        num_cols = df_prev.select_dtypes(include=[np.number]).columns.tolist()
        num_cols = [col for col in num_cols if col not in ['SK_ID_CURR', 'SK_ID_PREV']]

        agg_funcs = ['mean', 'max', 'min', 'sum']
        agg_dict = {col: agg_funcs for col in num_cols}
        agg_dict['SK_ID_PREV'] = ['count']

        df_prev_agg = df_prev.groupby('SK_ID_CURR').agg(agg_dict)

        df_prev_agg.columns = [
            f'PREV_{col[0]}_{col[1].upper()}' if col[0] != 'SK_ID_PREV'
            else 'PREV_APPLICATION_COUNT'
            for col in df_prev_agg.columns
        ]

        return df_prev_agg.reset_index()
