# src/pipelines/sklearn_pipeline.py

from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from configs.processeed_features_config import NUMERICAL_FEATURES, CATEGORICAL_FEATURES, BIN_CATEGORICAL_FEATURES
from src.pipelines.base_pipeline import BasePipelineBuilder

from src.pipelines.base_preprocessor import (
    get_base_feature_engineering_steps,
    get_numerical_transformer,
    get_categorical_transformer,
    get_binary_transformer
)

class SKLearnPipelineBuilder(BasePipelineBuilder):
    """
    Строитель для стандартных Sklearn-моделей.
    Использует полный препроцессинг (OHE, Scaling).
    """

    def _get_preprocessor(self, feature_engineering: bool = True) -> Pipeline:
        """
        Собирает полный пайплайн препроцессинга.
        FE (если нужно) -> Sklearn Full Preprocessor.
        """
        return get_sklearn_preprocessing_pipeline(
            include_feature_engineering=feature_engineering
        )

    def _get_model(self) -> BaseEstimator:
        """
        Получает инстанс модели, используя self.random_state из Base.
        """
        params = {'random_state': self.random_state}
        params.update(self.model_params)

        # в реальном коде тут должна быть логика извлечения model_class
        return LogisticRegression(**params)


def get_sklearn_preprocessing_pipeline(include_feature_engineering: bool = True) -> Pipeline:
    """
    Создает полный пайплайн препроцессинга для стандартных Sklearn/XGBoost моделей.
    """
    # logger.info(f"Building SKLEARN/XGBoost pipeline (FE={'enabled' if include_feature_engineering else 'disabled'})")

    steps = []

    if include_feature_engineering:
        steps.extend(get_base_feature_engineering_steps())

    column_transformer = ColumnTransformer(
        transformers=[
            ('num', get_numerical_transformer(scaled=True), NUMERICAL_FEATURES),
            ('cat', get_categorical_transformer(), CATEGORICAL_FEATURES),
            ('bin', get_binary_transformer(), BIN_CATEGORICAL_FEATURES)
        ],
        remainder='drop',
        verbose_feature_names_out=False
    )

    steps.append(('preprocessor', column_transformer))
    return Pipeline(steps=steps)
