# src/pipelines/lightgbm_pipeline.py

from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline

from src.pipelines.base_pipeline import BasePipelineBuilder
from src.pipelines.preprocessor import get_model_specific_pipeline

class LightGBMPipelineBuilder(BasePipelineBuilder):
    """
    Cтроитель для LightGBMClassifier.
    Использует минимальный препроцессинг (только импутация).
    Логика FE берется из BasePipelineBuilder.
    """
    def _get_preprocessor(self, feature_engineering: bool = True) -> Pipeline:
        """
        собирает полный пайплайн препроцессинга: FE (если нужно) -> LGBM-препроцессор.
        """
        return get_model_specific_pipeline(
            model_name='lightgbm',
            include_feature_engineering=feature_engineering
        )


    def _get_model(self) -> BaseEstimator:
        """
        создает инстанс LGBMClassifier, используя self.random_state из Base.
        """
        params = {'random_state': self.random_state}
        params.update(self.model_params)

        return LGBMClassifier(**params)