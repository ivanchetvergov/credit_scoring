# src/preprocessors/base_prep.py

"""
Содержит базовые, переиспользуемые трансформеры (num, cat, bin)
и стандартные шаги Feature Engineering.
"""
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from src.features.custom_transformers import (
    AnomalyHandler,
    FeatureCreator,
    DataFrameCoercer
)
from src.config import MAX_CATEGORIES_FOR_OHE

# ==============================================================================
# БАЗОВЫЕ КОМПОНЕНТЫ
# ==============================================================================

def get_numerical_transformer(scaled: bool = True) -> Pipeline:
    """Пайплайн для числовых признаков: импутация медианой, опционально масштабирование."""
    steps = [('imputer', SimpleImputer(strategy='median'))]
    if scaled:
        steps.append(('scaler', StandardScaler()))
    return Pipeline(steps=steps)


def get_categorical_transformer(max_categories: int = None) -> Pipeline:
    """Пайплайн для категориальных признаков: импутация 'missing' -> OHE."""
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


def get_categorical_type_transformer() -> Pipeline:
    """Пайплайн для категориальных признаков: импутация -> приведение к 'category'."""
    # Используем OrdinalEncoder, чтобы превратить строки в числа,

    return Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(
            handle_unknown='ignore',
            sparse_output=False,
        ))
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