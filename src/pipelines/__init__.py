# src/pipelines/__init__.py
from sklearn.pipeline import Pipeline

from src.pipelines.sklearn_pipeline import get_sklearn_preprocessing_pipeline
from src.pipelines.catboost_pipeline import get_catboost_preprocessing_pipeline
from src.pipelines.lightgbm_pipeline import get_lightgbm_preprocessing_pipeline
from src.pipelines.sklearn_pipeline import get_sklearn_preprocessing_pipeline
from src.pipelines.xgboost_pipeline import get_xgboost_preprocessing_pipeline


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

    elif model_name_lower == 'xgboost':
        return get_xgboost_preprocessing_pipeline(include_feature_engineering)

    elif model_name_lower in ('logistic_regression', 'random_forest', 'sgd_classifier'):
        return get_sklearn_preprocessing_pipeline(include_feature_engineering)

    else:
        # Для всех остальных по умолчанию используем полный пайплайн
        return get_sklearn_preprocessing_pipeline(include_feature_engineering)
