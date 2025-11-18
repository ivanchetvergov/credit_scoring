# src/features/custom_transformers.py
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import logging
from sklearn.model_selection import KFold
from sklearn.base import BaseEstimator, TransformerMixin

from src.config import SEED, BIN_CATEGORICAL_FEATURES

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


class FeatureCreator(BaseEstimator, TransformerMixin):
    """
        Создает продвинутые признаки из raw data.
        - Temporal: возраст, стаж, давность событий
        - Ratios: burden indicators (кредит/доход, аннуитет/доход)
        - Domain: credit risk clusters, age groups
        - Anomaly handling: DAYS_EMPLOYED sentinel
        - Interactions: квадраты ключевых фич
        """

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

        # --- 1. Temporal features ---
        if 'DAYS_BIRTH' in X.columns:
            X['AGE_YEARS'] = (-X['DAYS_BIRTH'] / 365).round(1)
            X['AGE_GROUP'] = pd.cut(X['AGE_YEARS'], bins=[0, 25, 35, 45, 55, 100], labels=[1, 2, 3, 4, 5])
            self.created_features_.extend(['AGE_YEARS', 'AGE_GROUP'])

        if 'DAYS_EMPLOYED' in X.columns:
            # Handle sentinel
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

        # --- 2. Credit burden indicators ---
        if 'AMT_CREDIT' in X.columns and 'AMT_INCOME_TOTAL' in X.columns:
            X['CREDIT_INCOME_RATIO'] = X['AMT_CREDIT'] / (X['AMT_INCOME_TOTAL'] + self.epsilon)
            # Cluster high/medium/low burden
            X['CREDIT_BURDEN_GROUP'] = pd.cut(
                X['CREDIT_INCOME_RATIO'],
                bins=[0, 5, 10, 50],
                labels=['low', 'medium', 'high']
            )
            self.created_features_.extend(['CREDIT_INCOME_RATIO', 'CREDIT_BURDEN_GROUP'])

        if 'AMT_ANNUITY' in X.columns and 'AMT_INCOME_TOTAL' in X.columns:
            X['ANNUITY_INCOME_RATIO'] = X['AMT_ANNUITY'] / (X['AMT_INCOME_TOTAL'] + self.epsilon)
            self.created_features_.append('ANNUITY_INCOME_RATIO')

        if 'AMT_ANNUITY' in X.columns and 'AMT_CREDIT' in X.columns:
            X['ANNUITY_CREDIT_RATIO'] = X['AMT_ANNUITY'] / (X['AMT_CREDIT'] + self.epsilon)
            # Payment term estimate
            X['CREDIT_TERM_EST'] = X['AMT_CREDIT'] / (X['AMT_ANNUITY'] + self.epsilon)
            self.created_features_.extend(['ANNUITY_CREDIT_RATIO', 'CREDIT_TERM_EST'])

        if 'AMT_GOODS_PRICE' in X.columns and 'AMT_CREDIT' in X.columns:
            X['GOODS_PRICE_CREDIT_RATIO'] = X['AMT_GOODS_PRICE'] / (X['AMT_CREDIT'] + self.epsilon)
            # Down payment proxy
            X['DOWN_PAYMENT_PROXY'] = (X['AMT_CREDIT'] - X['AMT_GOODS_PRICE']).clip(lower=0)
            self.created_features_.extend(['GOODS_PRICE_CREDIT_RATIO', 'DOWN_PAYMENT_PROXY'])

        # --- 3. Income per capita ---
        if 'AMT_INCOME_TOTAL' in X.columns and 'CNT_FAM_MEMBERS' in X.columns:
            X['INCOME_PER_PERSON'] = X['AMT_INCOME_TOTAL'] / (X['CNT_FAM_MEMBERS'] + self.epsilon)
            self.created_features_.append('INCOME_PER_PERSON')

        # --- 4. External source combinations (EXT_SOURCE are predictive) ---
        ext_cols = [c for c in ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3'] if c in X.columns]
        if len(ext_cols) >= 2:
            X['EXT_SOURCE_MEAN'] = X[ext_cols].mean(axis=1)
            X['EXT_SOURCE_MIN'] = X[ext_cols].min(axis=1)
            X['EXT_SOURCE_MAX'] = X[ext_cols].max(axis=1)
            X['EXT_SOURCE_RANGE'] = X['EXT_SOURCE_MAX'] - X['EXT_SOURCE_MIN']
            self.created_features_.extend(['EXT_SOURCE_MEAN', 'EXT_SOURCE_MIN', 'EXT_SOURCE_MAX', 'EXT_SOURCE_RANGE'])

        # --- 5. Document submission count ---
        doc_cols = [c for c in X.columns if c.startswith('FLAG_DOCUMENT_')]
        if doc_cols:
            X['DOCUMENTS_SUBMITTED'] = X[doc_cols].sum(axis=1)
            self.created_features_.append('DOCUMENTS_SUBMITTED')

        # --- 6. Interaction squares (key predictive features) ---
        for col in ['CREDIT_INCOME_RATIO', 'ANNUITY_INCOME_RATIO', 'AGE_YEARS', 'EMPLOYMENT_YEARS']:
            if col in X.columns:
                X[f'{col}_SQ'] = X[col] ** 2
                self.created_features_.append(f'{col}_SQ')

        logger.debug(f"AdvancedFeatureCreator: {len(self.created_features_)} features")
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


class CVTargetEncoder(BaseEstimator, TransformerMixin):
    """
    CV-safe target encoder. During fit(X, y) computes out-of-fold encoded values
    and stores full-data mappings for transform. For transform, unseen categories
    fallback to global target mean.
    Parameters:
    - n_splits: folds for OOF encoding
    - cols: optional list of columns to encode (if None, use all object/categorical)
    """

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
            # nothing to encode
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
                # handle NaNs by treating them as a category
                tr_series = X_tr[c].fillna('__NA__')
                val_series = X_val[c].fillna('__NA__')
                tr_means = y_tr.groupby(tr_series).mean()
                oof_vals = val_series.map(tr_means).fillna(self.global_mean_).values
                oof_encoded[c][val_idx] = oof_vals

        # store full-data mapping (fillna key for NaNs)
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


class IdentityTransformer(BaseEstimator, TransformerMixin):
    """Сериализуемый трансформер-заглушка. Возвращает входные данные."""
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X

