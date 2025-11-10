# src/models/sklearn_trainer.py
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier

from src.models.trainer_interface import BaseTrainer

class SklearnTrainer(BaseTrainer):
    """"
    Класс "учитель" для классических моделей Sklearn (LR, RF, SGD, ...)
    """

    SUPPORTED_MODELS = {
        'logistic_regression': LogisticRegression,
        'random_forest': RandomForestClassifier,
        'sgd_classifier': SGDClassifier
    }

    def __init__(self, model_name: str, **kwargs):
        if model_name not in self.SUPPORTED_MODELS:
            raise ValueError(f'Model {model_name} not supported. '
                             f'Supported model: {list(self.SUPPORTED_MODELS.keys())}')
        # инициализируем родительский класс, передавая имя модели
        super().__init__(model_name=model_name, **kwargs)

    def _get_model(self):
        """
        Реализация абстрактного метода: возвращает инициализированный LGBMClassifier.
        """
        model_class = self.SUPPORTED_MODELS[self.model_name]

        return model_class(**self.model_params)