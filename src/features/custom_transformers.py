# src/features/custom_transformers.py
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
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
            X['CREDIT_INCOME_RATIO'] = X['AMT_CREDIT'] / (X['AMT_INCOME_TOTAL'] + self.epsilon)
            self.created_features_.append('CREDIT_INCOME_RATIO')

        # 2 отношение аннуитета к доходу (payment burden)
        if 'AMT_ANNUITY' in X.columns and 'AMT_INCOME_TOTAL' in X.columns:
            X['ANNUITY_INCOME_RATIO'] = X['AMT_ANNUITY'] / (X['AMT_INCOME_TOTAL'] + self.epsilon)
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
            X['GOODS_PRICE_TO_CREDIT_RATIO'] = X['AMT_GOODS_PRICE'] / (X['AMT_CREDIT'] + self.epsilon)
            self.created_features_.append('GOODS_PRICE_TO_CREDIT_RATIO')

        logger.debug(f"Feature creation complete. Created {len(self.created_features_)} features.")

        return X


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

    def fit(self, X: pd.DataFrame, y=None):
        """
        Предвычисляет агрегации на этапе fit.
        Это позволяет избежать data leakage между train и test.
        """
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
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Применяет предвычисленные агрегации через merge.
        """
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

    @staticmethod
    def _aggregate_bureau(df_bureau: pd.DataFrame, df_bb_agg: pd.DataFrame) -> pd.DataFrame:
        """
        Агрегирует данные о предыдущих кредитах из Bureau на уровень клиента.
        Также мержит агрегации из Bureau Balance.

        :param df_bureau: Bureau DataFrame
        :param df_bb_agg: Aggregated Bureau Balance DataFrame
        :return: aggregated Bureau features
        """
        logger.debug("Aggregating Bureau features...")

        if df_bureau.empty:
            logger.warning("Bureau DataFrame is empty")
            return pd.DataFrame()

        try:
            # --- 1. мержим с Bureau Balance агрегациями ---
            if not df_bb_agg.empty:
                df_bureau = df_bureau.merge(df_bb_agg, on='SK_ID_BUREAU', how='left')
                logger.debug(f"Merged Bureau Balance into Bureau. Shape: {df_bureau.shape}")

            # --- 2. выбираем числовые колонки для агрегации ---
            num_cols = df_bureau.select_dtypes(include=[np.number]).columns.tolist()
            # исключаем ID колонки
            num_cols = [col for col in num_cols if col not in ['SK_ID_CURR', 'SK_ID_BUREAU']]

            # --- 3. агрегируем на уровень клиента ---
            agg_funcs = ['mean', 'max', 'min', 'sum']
            agg_dict = {col: agg_funcs for col in num_cols}
            # считаем количество кредитов
            agg_dict['SK_ID_BUREAU'] = ['count']

            df_bureau_agg = df_bureau.groupby('SK_ID_CURR').agg(agg_dict)

            # --- 4. переименовываем колонки ---
            df_bureau_agg.columns = [
                f'BUREAU_{col[0]}_{col[1].upper()}' if col[0] != 'SK_ID_BUREAU'
                else 'BUREAU_CREDIT_COUNT'
                for col in df_bureau_agg.columns
            ]

            logger.debug(f"Bureau aggregation complete. Created {len(df_bureau_agg.columns)} features")

            return df_bureau_agg.reset_index()

        except Exception as e:
            logger.error(f"Error aggregating Bureau: {e}", exc_info=True)
            raise

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
        # определяем нечисловые колонки для защиты от коэрции
        self.non_numeric_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

        # добавляем бинарные, которые могут быть 'Y'/'N'
        from src.config import BIN_CATEGORICAL_FEATURES
        self.non_numeric_cols.extend(BIN_CATEGORICAL_FEATURES)
        self.non_numeric_cols = list(set(self.non_numeric_cols))
        return self

    def transform(self, X):
        X_out = X.copy()

        # перебираем все колонки и коэрцируем те, которые не в списке
        for col in X_out.columns:
            if col not in self.non_numeric_cols:
                # попытка преобразовать в число, 'coerce' заменяет ошибки на NaN
                X_out[col] = pd.to_numeric(X_out[col], errors='coerce')

            # отдельно обрабатываем бинарные, которые могут быть строками
            elif col in X_out.columns:
                X_out[col] = (
                    X_out[col]
                    .replace({'Y': 1, 'N': 0, 'y': 1, 'n': 0})
                )

        return X_out

class IdentityTransformer(BaseEstimator, TransformerMixin):
    """Сериализуемый трансформер-заглушка. Возвращает входные данные."""
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X



