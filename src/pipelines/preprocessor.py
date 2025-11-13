# src/pipelines/preprocessor.py
"""
Модуль для создания preprocessing пайплайнов.
Содержит фабричные функции для различных типов моделей.
"""
import logging
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer

from src.features.custom_transformers import (
    AnomalyHandler,
    FeatureCreator,
    DataFrameCoercer
)
from src.config import (
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    BIN_CATEGORICAL_FEATURES,
    MAX_CATEGORIES_FOR_OHE
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ==============================================================================
# БАЗОВЫЕ КОМПОНЕНТЫ ПАЙПЛАЙНА
# ==============================================================================

def get_numerical_transformer() -> Pipeline:
    """
    Возвращает пайплайн для обработки числовых признаков.
    Стратегия: импутация медианой -> масштабирование (StandardScaler).

    :return: Pipeline для числовых фичей
    """
    return Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])


def get_categorical_transformer(max_categories: int = None) -> Pipeline:
    """
    Возвращает пайплайн для обработки категориальных признаков.
    Стратегия: импутация константой 'missing' -> One-Hot Encoding.

    :param max_categories: максимальное количество категорий для OHE
    :return: Pipeline для категориальных фичей
    """
    if max_categories is None:
        max_categories = MAX_CATEGORIES_FOR_OHE

    return Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(
            handle_unknown='ignore',
            sparse_output=False,
            drop='first',
            max_categories=max_categories
        ))
    ])


def get_binary_transformer() -> Pipeline:
    """
    Возвращает пайплайн для бинарных признаков (0/1 или Y/N).
    Стратегия: импутация наиболее частым значением.

    :return: Pipeline для бинарных фичей
    """
    return Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent'))
    ])

def get_binary_encoder() -> Pipeline:
    """
    Возвращает пайплайн для бинарных признаков, кодируя их в числа (0/1).
    Используется для моделей, которые требуют числовые данные (LightGBM/XGBoost/CatBoost).

    :return: Pipeline для бинарных фичей с кодированием
    """
    return Pipeline(steps=[
        # cначала импутация
        ('imputer', SimpleImputer(strategy='most_frequent', fill_value='missing')),
        # затем кодирование строк ('F', 'M') в числа (0, 1)
        # OrdinalEncoder безопаснее для ColumnTransformer
        ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
    ])

# ==============================================================================
# ПОЛНЫЕ PREPROCESSING ПАЙПЛАЙНЫ
# ==============================================================================

def get_preprocessing_pipeline(include_feature_engineering: bool = True) -> Pipeline:
    """
    Создает полный пайплайн препроцессинга для стандартных Sklearn моделей
    (Logistic Regression, Random Forest, SGD, XGBoost).

    Структура:
    1. DataFrameCoercer - приведение типов
    2. AnomalyHandler - обработка аномальных значений
    3. FeatureCreator - создание новых фичей (если включен FE)
    4. ColumnTransformer - обработка числовых/категориальных/бинарных фичей

    :param include_feature_engineering: включать ли feature engineering шаги
    :return: полный Pipeline препроцессинга
    """
    logger.info(f"Building preprocessing pipeline (FE={'enabled' if include_feature_engineering else 'disabled'})")

    steps = []

    # --- 1. Feature Engineering ---
    if include_feature_engineering:
        steps.extend([
            ('coercer', DataFrameCoercer()),
            ('anomaly_handler', AnomalyHandler()),
            ('feature_creator', FeatureCreator()),
        ])

    # --- 2. Column Transformer  ---
    column_transformer = ColumnTransformer(
        transformers=[
            ('num', get_numerical_transformer(), NUMERICAL_FEATURES),
            ('cat', get_categorical_transformer(), CATEGORICAL_FEATURES),
            ('bin', get_binary_transformer(), BIN_CATEGORICAL_FEATURES)
        ],
        remainder='drop',
        verbose_feature_names_out=False
    )

    steps.append(('preprocessor', column_transformer))

    logger.info(f"Preprocessing pipeline created with {len(steps)} steps")
    return Pipeline(steps=steps)


def get_lightgbm_preprocessing_pipeline(include_feature_engineering: bool = True) -> Pipeline:
    """
    Создает минимальный пайплайн препроцессинга для LightGBM.
    LightGBM может работать с категориальными фичами напрямую, поэтому
    мы делаем только импутацию (без OHE и масштабирования).

    Структура:
    1. DataFrameCoercer - приведение типов
    2. AnomalyHandler - обработка аномальных значений
    3. FeatureCreator - создание новых фичей (если включен FE)
    4. ColumnTransformer - только импутация и кодирование бинарных

    :param include_feature_engineering: включать ли feature engineering шаги
    :return: Pipeline для LightGBM
    """
    logger.info(
        f"Building LightGBM preprocessing pipeline (FE={'enabled' if include_feature_engineering else 'disabled'})")

    steps = []

    # --- 1. Feature Engineering  ---
    if include_feature_engineering:
        steps.extend([
            ('coercer', DataFrameCoercer()),
            ('anomaly_handler', AnomalyHandler()),
            ('feature_creator', FeatureCreator()),
        ])

    # --- 2. Minimal ColumnTransformer  ---
    # Числовые: только медианная импутация
    num_pipe = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median'))
    ])

    # Категориальные (многозначные): только импутация константой (останутся строками для LGBM)
    cat_pipe = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
    ])

    # !!! ИСПРАВЛЕНИЕ: УДАЛЕНА СТАРАЯ КОНФЛИКТУЮЩАЯ ЛОГИКА cat_and_bin_features !!!
    column_transformer = ColumnTransformer(
        transformers=[
            ('num', num_pipe, NUMERICAL_FEATURES),
            # 1. Многозначные категории: только импутация (оставляем строками)
            ('cat', cat_pipe, CATEGORICAL_FEATURES),
            # 2. Бинарные: принудительно кодируются в числа (0/1)
            ('bin_enc', get_binary_encoder(), BIN_CATEGORICAL_FEATURES)
        ],
        remainder='drop',
        verbose_feature_names_out=False,
        # verbose=True
    )

    steps.append(('preprocessor', column_transformer))

    logger.info(f"LightGBM preprocessing pipeline created with {len(steps)} steps")
    return Pipeline(steps=steps)


def get_catboost_preprocessing_pipeline(include_feature_engineering: bool = True) -> Pipeline:
    """
    Создает минимальный пайплайн препроцессинга для CatBoost.
    CatBoost имеет встроенную обработку категориальных фичей и пропусков,
    поэтому нам нужен только Feature Engineering.

    Структура:
    1. DataFrameCoercer - приведение типов
    2. AnomalyHandler - обработка аномальных значений
    3. FeatureCreator - создание новых фичей (если включен FE)
    4. ColumnTransformer для обработки бинарных и импутация для числовых.

    :param include_feature_engineering: включать ли feature engineering шаги
    :return: Pipeline для CatBoost
    """
    logger.info(
        f"Building CatBoost preprocessing pipeline (FE={'enabled' if include_feature_engineering else 'disabled'})")

    # steps = []
    #
    # # --- Feature Engineering ---
    # if include_feature_engineering:
    #     steps.extend([
    #         ('coercer', DataFrameCoercer()),
    #         ('anomaly_handler', AnomalyHandler()),
    #         ('feature_creator', FeatureCreator()),
    #     ])
    #
    # # --- Minimal ColumnTransformer for CatBoost (УСТОЙЧИВАЯ ВЕРСИЯ) ---
    # # CatBoost обработает категориальные фичи сам, если они переданы через cat_features
    # # Мы только импутируем числовые и кодируем бинарные для безопасности.
    # num_passthrough = Pipeline(steps=[('imputer', SimpleImputer(strategy='median'))])
    #
    # # cat_passthrough: для категориальных (строковых) фичей. Passthrough - самый безопасный выбор.
    # cat_passthrough = 'passthrough'
    #
    #
    # column_transformer = ColumnTransformer(
    #     transformers=[
    #         ('num', num_passthrough, NUMERICAL_FEATURES),
    #         ('cat', cat_passthrough, CATEGORICAL_FEATURES),
    #         # Бинарные: принудительно кодируем в числа (0/1)
    #         ('bin_enc', get_binary_encoder(), BIN_CATEGORICAL_FEATURES)
    #     ],
    #     remainder='drop',
    #     verbose_feature_names_out=False
    # )
    #
    # steps.append(('preprocessor', column_transformer))

    logger.info(f"CatBoost preprocessing pipeline created with {len(steps)} steps")
    return Pipeline(steps=steps)


# ==============================================================================
# ФАБРИЧНАЯ ФУНКЦИЯ (FACTORY PATTERN)
# ==============================================================================

def get_model_specific_pipeline(model_name: str, include_feature_engineering: bool = True) -> Pipeline:
    """
    Фабричная функция для создания пайплайна, оптимизированного под конкретную модель.

    :param model_name: название модели ('lightgbm', 'catboost', 'xgboost', 'logistic_regression', и т.д.)
    :param include_feature_engineering: включать ли feature engineering
    :return: оптимизированный Pipeline для данной модели
    """
    model_name_lower = model_name.lower()

    if model_name_lower == 'lightgbm':
        return get_lightgbm_preprocessing_pipeline(include_feature_engineering)

    elif model_name_lower == 'catboost':
        return get_catboost_preprocessing_pipeline(include_feature_engineering)

    else:
        # для всех остальных (Sklearn, XGBoost) используем полный пайплайн
        return get_preprocessing_pipeline(include_feature_engineering)