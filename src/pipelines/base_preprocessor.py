# src/pipelines/base_preprocessor.py

"""
Содержит базовые, переиспользуемые трансформеры (num, cat, bin)
и стандартные шаги Feature Engineering.
"""
from typing import Optional, List, Sequence, Dict

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, PowerTransformer, RobustScaler
from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer

from src.features.transformers.cv_target_encoder import CVTargetEncoder
from src.features.transformers.dataframe_coercer import DataFrameCoercer
from src.features.transformers.feature_creator import FeatureCreator
from src.features.transformers.anomaly_handler import AnomalyHandler
from src.config import MAX_CATEGORIES_FOR_OHE, SEED


# ==============================================================================
# БАЗОВЫЕ КОМПОНЕНТЫ
# ==============================================================================

def get_numerical_transformer(scaled: bool = True,
                              use_pca: bool = False,
                              pca_variance: float = 0.95) -> Pipeline:
    """Пайплайн для числовых признаков: импутация медианой, опционально масштабирование."""
    steps = [('imputer', SimpleImputer(strategy='median')),
             ('power_transform', PowerTransformer(method='yeo-johnson'))]
    if scaled:
        steps.append(('scaler', RobustScaler()))
    if use_pca:
        steps.append(('pca', PCA(n_components=pca_variance, random_state=SEED)))

    return Pipeline(steps=steps)

def get_categorical_transformer(use_target_enc: bool = True,
                                max_categories: Optional[int] = MAX_CATEGORIES_FOR_OHE,
                                n_splits: int = 5) -> Pipeline:
    """Пайплайн для категориальных признаков: импутация -> приведение к 'category'."""
    # Используем OrdinalEncoder, чтобы превратить строки в числа,

    if use_target_enc:
        return Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('target_enc', CVTargetEncoder(n_splits=n_splits, random_state=SEED))
        ])
    else:
        return Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False,
                                     max_categories=max_categories))
        ])

def get_binary_transformer() -> Pipeline:
    """Пайплайн для бинарных признаков (без кодирования, только импутация)."""
    return Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent'))
    ])


def get_binary_encoder() -> Pipeline:
    """Пайплайн для бинарных признаков, кодируя их в числа (0/1)."""
    return Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent', fill_value='missing')),
        ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
    ])


def get_base_feature_engineering_steps() -> list:
    """Возвращает базовые шаги Feature Engineering (Coercer, Anomaly, Creator)."""
    return [
        ('coercer', DataFrameCoercer()),
        ('anomaly_handler', AnomalyHandler()),
        ('feature_creator', FeatureCreator()),
    ]


def build_column_preprocessor(numerical_cols: Sequence[str],
                              categorical_cols: Sequence[str],
                              binary_cols: Sequence[str],
                              use_target_enc: bool = True,
                              numeric_scaled: bool = True) -> ColumnTransformer:
    """
    Build a ColumnTransformer assembling numerical, categorical and binary pipelines.
    This gives a single preprocessor suitable for model training and is easier to
    extend/inspect than ad-hoc step lists.
    """
    transformers = [
        ('num', get_numerical_transformer(scaled=numeric_scaled), list(numerical_cols)),
        ('cat', get_categorical_transformer(use_target_enc=use_target_enc), list(categorical_cols)),
        ('bin', get_binary_transformer(), list(binary_cols)),
    ]
    return ColumnTransformer(transformers=transformers, remainder='passthrough')
