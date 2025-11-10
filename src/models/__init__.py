# src/models/__init__.py

# импортируем все конкретные тренеры для создания реестра
from .trainer_interface import BaseTrainer
from .catboost_trainer import CatBoostTrainer
from .lightgbm_trainer import LGBMTrainer
from .sklearn_trainer import SklearnTrainer
from .xgboost_trainer import XGBoostTrainer

# создаем словарь для сопоставления имени модели с классом тренера
# это позволяет скрипту train_model.py загружать их динамически
MODEL_TRAINERS_REGISTRY = {
    'logistic_regression': SklearnTrainer,
    'random_forest': SklearnTrainer,
    'sgd_classifier': SklearnTrainer,
    'lightgbm': LGBMTrainer,
    'catboost': CatBoostTrainer,
    'xgboost': XGBoostTrainer,
}

# добавляем список доступных моделей
AVAILABLE_MODELS = list(MODEL_TRAINERS_REGISTRY.keys())