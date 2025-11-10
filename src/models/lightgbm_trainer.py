# src/models/lightgpm_trainer.py
from lightgbm import LGBMClassifier

from src.models.trainer_interface import BaseTrainer

class LGBMTrainer(BaseTrainer):
    """"
    Класс "учитель" для классификатора LightGBM
    """
    def __init__(self, **kwargs):
        # инициализируем родительский класс, передавая имя модели
        super().__init__(model_name='lightgbm', **kwargs)

    def _get_model(self):
        """
        Реализация абстрактного метода: возвращает инициализированный LGBMClassifier.
        """
        return LGBMClassifier(**self.model_params)