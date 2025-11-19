# configs/sklearn_config
from src.config import SEED

CLASS_WEIGHT = {
    0 : 1,
    1 : 20
}

MODEL_PARAMS = {
    'logistic_regression': {
        'penalty': 'l2',
        'C': 0.5,              # более сильная регуляризация
        'solver': 'saga',
        'max_iter': 1000,
        'random_state': SEED,
    },
    'random_forest': {
        'n_estimators': 420,
        'max_depth': 5,           # уменьшенная глубина
        'min_samples_split': 200, # усиленная регуляризация
        'min_samples_leaf': 80,   # усиленная регуляризация
        'random_state': SEED,
        'n_jobs': -1,
        'verbose': 0,
        'max_features': 'sqrt',
        'bootstrap': True
    },
    'sgd_classifier': {
        'loss': 'modified_huber',
        'penalty': 'l2',
        'alpha': 0.0002,       # чуть сильнее регуляризация
        'max_iter': 1000,
        'random_state': SEED,
        'n_jobs': -1
    }
}