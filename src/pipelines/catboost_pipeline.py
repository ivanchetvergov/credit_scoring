# src/pipelines/catboost_pipeline.py (ОБНОВЛЕННЫЙ)

from catboost import CatBoostClassifier
from sklearn.pipeline import Pipeline

from src.features import IdentityTransformer
from src.pipelines.base_pipeline import BasePipelineBuilder
from configs.catboost_config import CAT_FEATURES_COLS

class CatBoostPipelineBuilder(BasePipelineBuilder):
    """
    Строитель для CatBoostClassifier.
    Использует минимальный препроцессинг (только FE) с помощью фабричной функции.
    Инкапсулирует передачу списка категориальных фичей в CatBoostClassifier.
    """
    def _get_preprocessor(self, feature_engineering: bool = True) -> Pipeline:
        """
        Получает специфичный для CatBoost пайплайн препроцессинга,
        используя фабричную функцию.
        """
        return get_catboost_preprocessing_pipeline(
            include_feature_engineering=feature_engineering
        )

    def _get_model(self) -> CatBoostClassifier:
        """
        Создает инстанс CatBoostClassifier, используя self.random_state из Base.
        ВАЖНО: Инкапсулируем список категориальных фичей здесь.
        """
        params = {'random_state': self.random_state}
        params.update(self.model_params)

        # 1. определяем категориальные фичи, которые CatBoost должен обработать
        cat_features_for_model = CAT_FEATURES_COLS

        # 2. Добавляем их к параметрам инициализации модели
        params['cat_features'] = cat_features_for_model

        # verbose=False по умолчанию для чистоты логов
        return CatBoostClassifier(verbose=False, **params)


def get_catboost_preprocessing_pipeline(include_feature_engineering: bool = True) -> Pipeline:
    """
    Создает минимальный Sklearn Pipeline, содержащий только IdentityTransformer.
    FE шаги должны быть применены ОТДЕЛЬНО в CatBoostTrainer.
    """
    # Пайплайн CatBoost состоит только из сериализуемой заглушки.
    return Pipeline(steps=[
        ('passthrough', IdentityTransformer()),
    ])
