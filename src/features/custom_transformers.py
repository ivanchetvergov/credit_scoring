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


class AuxiliaryFeatureAggregator(BaseEstimator, TransformerMixin):
    """
    Трансформер для агрегации данных из вспомогательных таблиц
    (Bureau, Previous Applications и т.д.) на уровень клиента (SK_ID_CURR).

    Fit: Не делает ничего.
    Transform: Выполняет агрегацию и слияние.
    """

    def __init__(self, aux_data: Dict[str, pd.DataFrame]):
        """
        :param aux_data: Словарь вспомогательных датафреймов.
                         Пример: {'bureau': df_bureau, 'bureau_balance': df_bb, ...}
        """
        self.aux_data = aux_data

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_out = X.copy()

        # --- 1. агрегация Bureau Balance
        df_bb_agg = self._aggregate_bureau_balance(self.aux_data.get('bureau_balance', pd.DataFrame()))

        # --- 2. агрегация Bureau (передаем bureau и результат bb_agg)
        df_bureau_agg = self._aggregate_bureau(
            self.aux_data.get('bureau', pd.DataFrame()),
            df_bb_agg
        )

        # --- 3. агрегация Previous Application
        df_prev_agg = self._aggregate_previous_application(self.aux_data.get('previous_application', pd.DataFrame()))

        # --- 4. финальное слияние
        if not df_bureau_agg.empty:
            X_out = X_out.merge(df_bureau_agg, on='SK_ID_CURR', how='left')
            logger.debug(f"Merged Bureau features. Shape: {X_out.shape}")

        if not df_prev_agg.empty:
            X_out = X_out.merge(df_prev_agg, on='SK_ID_CURR', how='left')
            logger.debug(f"Merged Previous App features. Shape: {X_out.shape}")

        return X_out

    # --- внутренние методы ---
    @staticmethod
    def _aggregate_bureau_balance(self, df_bb: pd.DataFrame) -> pd.DataFrame:
        """
            Агрегирует ежемесячные статусы просрочки из bureau_balance на уровень
            каждого предыдущего кредита (SK_ID_BUREAU)

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
    def _aggregate_bureau(self, df_bureau: pd.DataFrame) -> pd.DataFrame:
        """
        Агрегирует ежемесячные статусы просрочки из bureau_balance на уровень
        каждого предыдущего кредита (SK_ID_BUREAU)

        :param df_bureau: Bureau Balance DataFrame
        :return: aggregated Bureau Balance DataFrame
        """
        logger.debug("Aggregating Bureau Balance features...")

        # проверка на пустой датафрейм
        if df_bureau.empty:
            logger.warning("Bureau Balance DataFrame is empty")
            return pd.DataFrame()

        try:
            # --- 1 считаем частоту каждого статуса просрочки
            df_bureau_agg = (
                df_bureau
                .groupby('SK_ID_BUREAU')['STATUS']
                .value_counts(normalize=True)
                .unstack(fill_value=0)
            )

            # --- 2 переименовываем колонки
            df_bureau_agg.columns = [f'BB_STATUS_PCT_{col}' for col in df_bureau_agg.columns]

            # --- 3 добавляем длину кредитной истории
            df_bureau_agg['BB_HISTORY_LENGTH'] = df_bureau.groupby('SK_ID_BUREAU').size()

            logger.debug(f"Bureau Balance features created. Shape: {df_bureau_agg.shape}")
            return df_bureau_agg.reset_index()

        except Exception as e:
            logger.error(f"Error aggregating Bureau Balance: {e}", exc_info=True)
            raise

    @staticmethod
    def _aggregate_previous_application(self, df_prev: pd.DataFrame) -> pd.DataFrame:
        """
           Агрегирует данные о предыдущих заявках на уровень клиента

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
                    # .fillna(0)  # заполнение пропусков в бинарных
                )
        return X_out
