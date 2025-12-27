# src/pipelines/preprocessing.py
"""Preprocessing pipelines для всех моделей."""
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, PowerTransformer, RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin

from src.config import NUMERICAL_FEATURES, CATEGORICAL_FEATURES, BIN_CATEGORICAL_FEATURES, SEED
from src.features.transformers import DataFrameCoercer, FeatureCreator
from src.features.transformers.cv_target_encoder import CVTargetEncoder


def get_numerical_transformer(scaled: bool = True) -> Pipeline:
    steps = [
        ('imputer', SimpleImputer(strategy='median')),
        ('power', PowerTransformer(method='yeo-johnson')),
    ]
    if scaled:
        steps.append(('scaler', RobustScaler()))
    return Pipeline(steps)


def get_categorical_transformer(use_target_enc: bool = True) -> Pipeline:
    if use_target_enc:
        return Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('encoder', CVTargetEncoder(n_splits=5, random_state=SEED))
        ])
    return Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])


def get_binary_transformer() -> Pipeline:
    return Pipeline([('imputer', SimpleImputer(strategy='most_frequent'))])


def get_feature_engineering_steps() -> list:
    return [
        ('coercer', DataFrameCoercer()),
        ('feature_creator', FeatureCreator()),
    ]


class CatBoostPreprocessor(BaseEstimator, TransformerMixin):
    """
    Специализированный preprocessor для CatBoost.
    Применяет FE и подготавливает категориальные колонки (добавляет 'Missing' в categories).
    """

    def __init__(self):
        self.fe_pipeline = Pipeline(get_feature_engineering_steps())

    def fit(self, X, y=None):
        self.fe_pipeline.fit(X, y)
        return self

    def transform(self, X):
        X_out = self.fe_pipeline.transform(X)
        if not isinstance(X_out, pd.DataFrame):
            X_out = pd.DataFrame(X_out)

        # Подготовка категориальных колонок для CatBoost
        cat_cols = X_out.select_dtypes(include=['object', 'category']).columns.tolist()
        for col in cat_cols:
            series = X_out[col].astype('category')
            if 'Missing' not in series.cat.categories:
                series = series.cat.add_categories(['Missing'])
            X_out[col] = series.fillna('Missing')

        return X_out


def get_preprocessing_pipeline(model_name: str) -> Pipeline:
    """
    Фабрика preprocessing pipelines.

    CatBoost: FE + категориальная подготовка (без ColumnTransformer)
    Остальные: FE + ColumnTransformer (impute, scale, encode)
    """
    model_name = model_name.lower()

    if model_name == 'catboost':
        return Pipeline([('catboost_prep', CatBoostPreprocessor())])

    # Для sklearn, LightGBM, XGBoost — полный pipeline
    steps = get_feature_engineering_steps()

    column_transformer = ColumnTransformer(
        transformers=[
            ('num', get_numerical_transformer(scaled=True), NUMERICAL_FEATURES),
            ('cat', get_categorical_transformer(use_target_enc=True), CATEGORICAL_FEATURES),
            ('bin', get_binary_transformer(), BIN_CATEGORICAL_FEATURES),
        ],
        remainder='drop',
        verbose_feature_names_out=False
    )

    steps.append(('preprocessor', column_transformer))
    return Pipeline(steps)

