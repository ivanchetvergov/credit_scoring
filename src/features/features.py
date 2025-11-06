import pandas as pd
import numpy as np
import logging
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

from src.config import (
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    BIN_CATEGORICAL_FEATURES
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

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        if 'DAYS_EMPLOYED' in X.columns:
            # индикатор аномалии (безработный/неизвестно)
            X['DAYS_EMPLOYED_ANOM'] = (X['DAYS_EMPLOYED'] == self.anomaly_value).astype(int)

            # заменяем аномальные значения на NaN
            X.loc[X['DAYS_EMPLOYED'] == self.anomaly_value, 'DAYS_EMPLOYED'] = np.nan

            logger.info(f"Anomaly handling: found {X['DAYS_EMPLOYED_ANOM'].sum()} anomalies")

        return X


class FeatureCreator(BaseEstimator, TransformerMixin):
    """
    Создает новые, полезные признаки на основе существующих.
    """
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    @staticmethod
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        # 1 отношение кредита к доходу (credit burden)
        if 'AMT_CREDIT' in X.columns and 'AMT_INCOME_TOTAL' in X.columns:
            X['CREDIT_INCOME_RATIO'] = X['AMT_CREDIT'] / (X['AMT_INCOME_TOTAL'] + 1)

        # 2 отношение аннуитета к доходу (payment burden)
        if 'AMT_ANNUITY' in X.columns and 'AMT_INCOME_TOTAL' in X.columns:
            X['ANNUITY_INCOME_RATIO'] = X['AMT_ANNUITY'] / (X['AMT_INCOME_TOTAL'] + 1)

        # 3 возраст клиента в годах
        if 'DAYS_BIRTH' in X.columns:
            X['AGE_YEARS'] = (X['DAYS_BIRTH'] / -365).round(1)

        # 4 стаж работы в годах
        if 'DAYS_EMPLOYED' in X.columns:
            X['EMPLOYMENT_YEARS'] = (X['DAYS_EMPLOYED'] / -365).clip(lower=0).round(1)

        # 5 отношение стоимости товара к кредиту
        if 'AMT_GOODS_PRICE' in X.columns and 'AMT_CREDIT' in X.columns:
            X['GOODS_PRICE_TO_CREDIT_RATIO'] = X['AMT_GOODS_PRICE'] / (X['AMT_CREDIT'] + 1)

        logger.info(f"Feature creation complete. New shape: {X.shape}")

        return X


class FeatureSelector(BaseEstimator, TransformerMixin):
    """
    Выбирает только нужные признаки для модели.
    """

    def __init__(self, feature_list):
        self.feature_list = feature_list

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        # проверяем какие признаки есть в данных
        available_features = [f for f in self.feature_list if f in X.columns]
        missing_features = [f for f in self.feature_list if f not in X.columns]

        if missing_features:
            logger.warning(f"Missing features: {missing_features[:5]}...")

        logger.info(f"Selected {len(available_features)} features out of {len(self.feature_list)}")

        return X[available_features]


# ========================== #
 # 2 пайплайн предобработки #
# ========================== #

def get_numerical_transformer() -> Pipeline:
    """
    Создает пайплайн для обработки числовых признаков.
    :return: Pipeline для числовых признаков
    """
    return Pipeline(
        steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]
    )


def get_categorical_transformer() -> Pipeline:
    """
    Создает пайплайн для обработки категориальных признаков.
    :return: Pipeline для категориальных признаков
    """
    return Pipeline(
        steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop='first'))
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
        # remainder='drop',  # удаляем все остальные колонки
        remainder=get_numerical_transformer(),
        verbose_feature_names_out=False
    )

    return preprocessor


def get_preprocessing_pipeline(with_feature_engineering=True) -> Pipeline:
    """
    Создает полный пайплайн препроцессинга для sklearn моделей.
    :param with_feature_engineering: Включать ли feature engineering
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


if __name__ == '__main__':
    from src.config import FEATURE_STORE_PATH, BASE_FEATURES

    logger.info("Testing preprocessing pipeline...")

    # загружаем данные

    df = pd.read_parquet(FEATURE_STORE_PATH)

    # используем только признаки, которые уже существуют в загруженном DataFrame ---
    present_base_features = [f for f in BASE_FEATURES if f in df.columns]

    # анализируем пропуски
    print("\n" + "=" * 80)
    print("MISSING VALUES ANALYSIS")
    print("=" * 80)

    # используем отфильтрованный список признаков:
    missing_df = analyze_missing_values(df[present_base_features])
    print(missing_df.head(20))

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