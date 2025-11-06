import pandas as pd
import numpy as np
from typing import List, Optional
import logging
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

from src.config import (
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    BIN_CATEGORICAL_FEATURES,
    MAX_CATEGORIES_FOR_OHE
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ========================== #
 # 1 кастомные трансформеры #
# ========================== #

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


class FeatureSelector(BaseEstimator, TransformerMixin):
    """
    Выбирает только нужные признаки для модели.

    :param feature_list: List[str] Список признаков для выбора
    :param raise_on_missing: bool Вызывать ли ошибку при отсутствии признаков (по умолчанию False)
    """

    def __init__(self, feature_list: List[str], raise_on_missing: bool = False):
        self.feature_list = feature_list
        self.raise_on_missing = raise_on_missing
        self.available_features_: List[str] = []
        self.missing_features_: List[str] = []

    def fit(self, X: pd.DataFrame, y=None) -> 'FeatureSelector':
        # проверяем какие признаки есть в данных
        self.available_features_ = [f for f in self.feature_list if f in X.columns]
        self.missing_features_ = [f for f in self.feature_list if f not in X.columns]

        if self.missing_features_:
            msg = f"Missing {len(self.missing_features_)} features: {self.missing_features_[:5]}"
            if self.raise_on_missing:
                raise ValueError(msg)
            else:
                logger.warning(msg)

        logger.debug(f"Selected {len(self.available_features_)} features out of {len(self.feature_list)}")

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X[self.available_features_]


# ========================== #
 # 2 пайплайн предобработки #
# ========================== #

def get_numerical_transformer() -> Pipeline:
    """
    Создает пайплайн для обработки числовых признаков.
    :return: Pipeline для числовых признаков: imputer -> scaler
    """
    return Pipeline(
        steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]
    )


def get_categorical_transformer(max_categories: Optional[int] = None) -> Pipeline:
    """
    Создает пайплайн для обработки категориальных признаков.
    :param max_categories: макс. кол-во категорий для OHE
    :return: Pipeline для категориальных признаков
    """
    if max_categories is None:
        max_categories = MAX_CATEGORIES_FOR_OHE

    return Pipeline(
        steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(
                handle_unknown='ignore',
                sparse_output=False,
                drop='first',
                max_categories=max_categories
            ))
        ]
    )


def get_binary_transformer() -> Pipeline:
    """
    Создает пайплайн для бинарных признаков.
    :return: Pipeline для бинарных признаков
    """
    return Pipeline(
        steps=[
            ('imputer', SimpleImputer(strategy='most_frequent'))
        ]
    )


def get_full_preprocessor() -> ColumnTransformer:
    """
    Создает ColumnTransformer для финальной предобработки данных.
    Используется для sklearn моделей.
    :return: ColumnTransformer
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', get_numerical_transformer(), NUMERICAL_FEATURES),
            ('cat', get_categorical_transformer(), CATEGORICAL_FEATURES),
            ('bin', get_binary_transformer(), BIN_CATEGORICAL_FEATURES)
        ],
        remainder='drop',  # удаляем все остальные колонки
        verbose_feature_names_out=False
    )

    return preprocessor


def get_preprocessing_pipeline(with_feature_engineering=True) -> Pipeline:
    """
    Создает полный пайплайн препроцессинга для sklearn моделей.
    :param with_feature_engineering: Включать ли feature engineering (default=True)
    :return: Pipeline
    """
    steps = []

    if with_feature_engineering:
        steps.extend([
            ('anomaly_handler', AnomalyHandler()),
            ('feature_creator', FeatureCreator())
        ])

    steps.append(('preprocessor', get_full_preprocessor()))

    pipeline = Pipeline(steps=steps)

    logger.info(f"Preprocessing pipeline created with {len(steps)} steps")

    return pipeline


def get_catboost_preprocessing_pipeline() -> Pipeline:
    """
    Создает упрощенный пайплайн для CatBoost (без OHE и scaling).
    CatBoost работает с категориальными признаками напрямую.
    :return: Pipeline
    """
    pipeline = Pipeline(
        steps=[
            ('anomaly_handler', AnomalyHandler()),
            ('feature_creator', FeatureCreator()),
            # CatBoost не требует нормализации и OHE
        ]
    )

    logger.info("CatBoost preprocessing pipeline created")

    return pipeline


# ========================== #
# 3 вспомогательные функции #
# ========================== #

def get_feature_names(preprocessor: ColumnTransformer, input_features: list) -> list:
    """
    Извлекает названия признаков после ColumnTransformer.
    :param preprocessor: Fitted ColumnTransformer
    :param input_features: Исходные названия признаков
    :return: Список названий признаков после трансформации
    """
    feature_names = []

    for name, transformer, features in preprocessor.transformers_:
        if name == 'remainder':
            continue

        if hasattr(transformer, 'get_feature_names_out'):
            names = transformer.get_feature_names_out(features)
        else:
            names = features

        feature_names.extend(names)

    return feature_names


def analyze_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Анализирует пропущенные значения в датасете.

    :param df: DataFrame для анализа
    :return: DataFrame со статистикой по пропускам
    """
    missing_stats = pd.DataFrame({
        'feature': df.columns,
        'missing_count': df.isnull().sum().values,
        'missing_pct': (df.isnull().sum() / len(df) * 100).values
    })

    missing_stats = missing_stats[missing_stats['missing_count'] > 0].sort_values(
        'missing_pct', ascending=False
    )

    return missing_stats


def validate_features(df: pd.DataFrame, required_features: List[str]) -> bool:
    """
    Проверяет наличие всех необходимых признаков в DataFrame.

    :param df: pd.DataFrame для проверки
    :param required_features: List[str] cписок необходимых признаков
    :return: bool. True если все признаки присутствуют, иначе False
    """
    missing = [f for f in required_features if f not in df.columns]

    if missing:
        logger.error(f"Missing required features: {missing}")
        return False

    logger.info(f"All {len(required_features)} required features are present")
    return True


if __name__ == '__main__':
    from src.config import FEATURE_STORE_PATH

    logger.info("Testing preprocessing pipeline...")

    try:
        # загружаем данные
        df = pd.read_parquet(FEATURE_STORE_PATH)
        logger.info(f"Loaded feature store with shape: {df.shape}")

        # анализируем пропуски
        print("\n" + "=" * 80)
        print("MISSING VALUES ANALYSIS")
        print("=" * 80)

        missing_df = analyze_missing_values(df, top_n=20)
        print(missing_df)

        # тестируем пайплайн
        pipeline = get_preprocessing_pipeline()

        print("\n" + "=" * 80)
        print("TESTING PIPELINE")
        print("=" * 80)

        # берем небольшую выборку для теста
        sample = df.head(1000)
        transformed = pipeline.fit_transform(sample)

        print(f"\nOriginal shape: {sample.shape}")
        print(f"Transformed shape: {transformed.shape}")
        print(f"\nPipeline test completed successfully!")

    except FileNotFoundError:
        logger.error(f"Feature store not found at {FEATURE_STORE_PATH}")
        logger.info("Please run feature_engineering.py first")
    except Exception as e:
        logger.error(f"Error during pipeline testing: {e}", exc_info=True)