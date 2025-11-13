# src/preprocessing/base_preprocessor
import logging
from typing import Optional
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

def get_numerical_transformer() -> Pipeline:
    """
    Возвращает пайплайн для обработки числовых признаков.
    imputer (median) -> scaler (standard).
    """
    return Pipeline(
        steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]
    )

def get_categorical_transformer(max_categories: Optional[int] = None) -> Pipeline:
    """
    Возвращает пайплайн для OHE категориальных признаков.
    imputer (constant 'missing') -> encoder (OHE).
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
    Возвращает пайплайн для бинарных признаков (только импутация).
    """
    return Pipeline(
        steps=[
            ('imputer', SimpleImputer(strategy='most_frequent'))
        ]
    )

def get_full_preprocessor() -> ColumnTransformer:
    """
    Создает ColumnTransformer для финальной предобработки данных.
    используется Sklearn-моделями (LR, RF) и XGBoost.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', get_numerical_transformer(), NUMERICAL_FEATURES),
            ('cat', get_categorical_transformer(), CATEGORICAL_FEATURES),
            ('bin', get_binary_transformer(), BIN_CATEGORICAL_FEATURES)
        ],
        remainder='drop',
        verbose_feature_names_out=False
    )

    return preprocessor