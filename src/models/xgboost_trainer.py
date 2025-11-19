# src/models/xgboost_trainer.py (Обновленный)
from src.models.trainer_interface import BaseTrainer
from xgboost import XGBClassifier
from typing import Dict, Optional
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class XGBoostTrainer(BaseTrainer):
    """"
    Класс "учитель" для XGBoostClassifier.
    Использует полный пайплайн Sklearn (как и другие обычные модели),
    но добавляет аргументы для ранней остановки (early_stopping).
    """

    def __init__(self, **kwargs):
        # инициализируем родительский класс
        super().__init__(model_name='xgboost', **kwargs)

    def _get_model(self):
        """Возвращает инициализированный XGBClassifier."""
        return XGBClassifier(**self.model_params)

    def _build_fit_kwargs(
            self,
            fit_kwargs: Optional[Dict],
            *,
            X_train,
            y_train,
            X_test,
            y_test,
            preprocessor
    ) -> Dict:
        return dict(fit_kwargs or {})
