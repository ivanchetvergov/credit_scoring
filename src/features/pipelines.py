# src/features/pipelines.py
import pandas as pd
from typing import Optional
import logging
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

from src.features.custom_transformers import AnomalyHandler, FeatureCreator, DataFrameCoercer

from src.config import (
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    BIN_CATEGORICAL_FEATURES,
    MAX_CATEGORIES_FOR_OHE
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ========================== #
 # пайплайн предобработки #
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
            ('coercer_pre', DataFrameCoercer()),
            ('anomaly_handler', AnomalyHandler()),
            ('feature_creator', FeatureCreator()),
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
            ('coercer_pre', DataFrameCoercer()),
            ('anomaly_handler', AnomalyHandler()),
            ('feature_creator', FeatureCreator()),
            # CatBoost не требует нормализации и OHE
        ]
    )

    logger.info("CatBoost preprocessing pipeline created")

    return pipeline

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