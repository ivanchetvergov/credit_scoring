# src/pipelines/sklearn_pipeline.py

from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from src.pipelines.base_pipeline import BasePipelineBuilder
from src.pipelines.preprocessor import get_model_specific_pipeline

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
        return get_model_specific_pipeline(
            model_name= 'default',
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