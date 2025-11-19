# src/pipelines/xgboost_pipeline.py
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from xgboost.sklearn import XGBClassifier
from sklearn.base import BaseEstimator

from src.config import NUMERICAL_FEATURES, CATEGORICAL_FEATURES, BIN_CATEGORICAL_FEATURES
from src.pipelines.base_pipeline import BasePipelineBuilder
from src.pipelines.base_preprocessor import (
    get_base_feature_engineering_steps,
    get_numerical_transformer,
    get_binary_transformer
)

# ALL_SELECTED_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES + BIN_CATEGORICAL_FEATURES


class XGBoostPipelineBuilder(BasePipelineBuilder):
    """
    Строитель для XGBoostClassifier.
    использует полный препроцессинг (OHE, Scaling).
    """

    def _get_preprocessor(self, feature_engineering: bool = True) -> Pipeline:
        return get_xgboost_preprocessing_pipeline(
            include_feature_engineering=feature_engineering
        )

    def _get_model(self) -> BaseEstimator:
        params = {'random_state': self.random_state}
        params.update(self.model_params)

        return XGBClassifier(**params)


def get_xgboost_preprocessing_pipeline(include_feature_engineering: bool = True) -> Pipeline:
    """
    Создает минимальный пайплайн препроцессинга для XGBoost,
    оставляя категориальные признаки для нативной обработки.
    """

    steps = []

    if include_feature_engineering:
        steps.extend(get_base_feature_engineering_steps())

    # steps.append(('feature_selector', FeatureSelector(feature_list=ALL_SELECTED_FEATURES)))

    cat_pipe = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
    ])

    column_transformer = ColumnTransformer(
        transformers=[
            ('num', get_numerical_transformer(scaled=True), NUMERICAL_FEATURES),
            ('cat', cat_pipe, CATEGORICAL_FEATURES),
            ('bin', get_binary_transformer(), BIN_CATEGORICAL_FEATURES)
        ],
        remainder='drop',
        verbose_feature_names_out=False
    )

    steps.append(('preprocessor', column_transformer))
    return Pipeline(steps=steps)
