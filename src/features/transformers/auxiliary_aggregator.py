from typing import Optional, Dict

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from . import logger


class AuxiliaryFeatureAggregator(BaseEstimator, TransformerMixin):
    """
    Трансформер для агрегации данных из вспомогательных таблиц
    (Bureau, Previous Applications и т.д.) на уровень клиента (SK_ID_CURR).

    Fit: Предвычисляет агрегации на train данных.
    Transform: Применяет сохраненные агрегации через merge.
    """

    def __init__(self, aux_data: Optional[Dict[str, pd.DataFrame]] = None):
        """
        :param aux_data: Словарь вспомогательных датафреймов.
                         Пример: {'bureau': df_bureau, 'bureau_balance': df_bb, ...}
        """
        self.aux_data = aux_data or {}
        self.bureau_agg_ = None
        self.prev_agg_ = None
        # list of aggregation functions used for numeric columns
        self.agg_funcs = ['mean', 'max', 'min', 'sum', 'std', self._q25, self._q75]

    @staticmethod
    def _q25(x):
        return x.quantile(0.25)

    @staticmethod
    def _q75(x):
        return x.quantile(0.75)

    def fit(self, X: pd.DataFrame, y=None):
        logger.info("Fitting AuxiliaryFeatureAggregator...")

        # --- 1. агрегация Bureau Balance ---
        df_bb = self.aux_data.get('bureau_balance', pd.DataFrame())
        df_bb_agg = self._aggregate_bureau_balance(df_bb)

        # --- 2. агрегация Bureau ---
        df_bureau = self.aux_data.get('bureau', pd.DataFrame())
        self.bureau_agg_ = self._aggregate_bureau(df_bureau, df_bb_agg)

        # --- 3. агрегация Previous Application ---
        df_prev = self.aux_data.get('previous_application', pd.DataFrame())
        self.prev_agg_ = self._aggregate_previous_application(df_prev)

        logger.info("AuxiliaryFeatureAggregator fitted successfully")
        """
        Применяет предвычисленные агрегации через merge.
        """
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        X_out = X.copy()

        # --- финальное слияние ---
        if self.bureau_agg_ is not None and not self.bureau_agg_.empty:
            X_out = X_out.merge(self.bureau_agg_, on='SK_ID_CURR', how='left')
            logger.debug(f"Merged Bureau features. Shape: {X_out.shape}")

        if self.prev_agg_ is not None and not self.prev_agg_.empty:
            X_out = X_out.merge(self.prev_agg_, on='SK_ID_CURR', how='left')
            logger.debug(f"Merged Previous App features. Shape: {X_out.shape}")

        return X_out

    # --- внутренние методы агрегации ---

    @staticmethod
    def _aggregate_bureau_balance(df_bb: pd.DataFrame) -> pd.DataFrame:
        """
        Агрегирует ежемесячные статусы просрочки из bureau_balance на уровень
        каждого предыдущего кредита (SK_ID_BUREAU).

        :param df_bb: Bureau Balance DataFrame
        :return: aggregated Bureau Balance DataFrame
        """
        logger.debug("Aggregating Bureau Balance features...")

        # проверка на пустой датафрейм
        if df_bb.empty:
            logger.warning("Bureau Balance DataFrame is empty")
            return pd.DataFrame()

        try:
            # --- 1 считаем частоту каждого статуса просрочки
            df_bb_agg = (
                df_bb
                .groupby('SK_ID_BUREAU')['STATUS']
                .value_counts(normalize=True)
                .unstack(fill_value=0)
            )

            # --- 2 переименовываем колонки
            df_bb_agg.columns = [f'BB_STATUS_PCT_{col}' for col in df_bb_agg.columns]

            # --- 3 добавляем длину кредитной истории
            df_bb_agg['BB_HISTORY_LENGTH'] = df_bb.groupby('SK_ID_BUREAU').size()

            logger.debug(f"Bureau Balance features created. Shape: {df_bb_agg.shape}")
            return df_bb_agg.reset_index()

        except Exception as e:
            logger.error(f"Error aggregating Bureau Balance: {e}", exc_info=True)
            raise

    def _aggregate_bureau(self, df_bureau: pd.DataFrame, df_bb_agg: pd.DataFrame) -> pd.DataFrame:
        """
        Агрегирует ежемесячные статусы просрочки из bureau_balance на уровень
        каждого предыдущего кредита (SK_ID_BUREAU).

        :param df_bureau: Bureau Balance DataFrame
        :return: aggregated Bureau Balance DataFrame
        """
        logger.debug("Aggregating Bureau Balance features...")

        # проверка на пустой датафрейм
        if df_bureau.empty:
            logger.warning("Bureau Balance DataFrame is empty")
            return pd.DataFrame()

        if not df_bb_agg.empty:
            df_bureau = df_bureau.merge(df_bb_agg, on='SK_ID_BUREAU', how='left')

        num_cols = [
            c for c in df_bureau.select_dtypes(include=[np.number]).columns
            if c not in ['SK_ID_CURR', 'SK_ID_BUREAU']
        ]

        # Add percentiles + std
        agg_dict = {col: self.agg_funcs for col in num_cols}
        agg_dict['SK_ID_BUREAU'] = ['count']

        df_agg = df_bureau.groupby('SK_ID_CURR').agg(agg_dict)

        # Rename columns
        new_cols = []
        for col in df_agg.columns:
            col_name, func = col[0], col[1]
            if col_name == 'SK_ID_BUREAU':
                # count of bureau records per customer
                new_cols.append('BUREAU_CREDIT_COUNT')
                continue

            # determine function label
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
        """
        Агрегирует данные о предыдущих заявках на уровень клиента.

        :param df_prev: Previous Application DataFrame
        :return: Aggregated Previous Application features
        """
        logger.debug("Aggregating Previous Application features...")

        if df_prev.empty:
            logger.warning("Previous Application DataFrame is empty")
            return pd.DataFrame()

        try:
            # выбираем числовые колонки
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

            logger.debug(f"Previous Application aggregation complete. Created {len(df_prev_agg.columns)} features")

            return df_prev_agg.reset_index()

        except Exception as e:
            logger.error(f"Error aggregating Previous Application: {e}", exc_info=True)
            raise
