# configs/sklearn_config
from src.config import SEED

CLASS_WEIGHT = {
    0 : 1,
    1 : 20
}

MODEL_PARAMS = {
    'logistic_regression': {
        'penalty': 'l1',
        'C': 0.5,
        'solver': 'saga', # рекомендуется для L1 и больших данных
        'max_iter': 1000,
        'random_state': SEED,
        'class_weight': CLASS_WEIGHT
    },
    'random_forest': {
        'n_estimators': 420,
        'max_depth': 12,
        'min_samples_split': 60,
        'min_samples_leaf': 20,
        'random_state': SEED,
        'class_weight': CLASS_WEIGHT,
        'n_jobs': -1,
        'verbose': 0,
        'max_features': 'sqrt',
        'bootstrap': True
    },
    'sgd_classifier': {
        'loss': 'modified_huber',
        'penalty': 'l2',
        'alpha': 0.0001,
        'max_iter': 1000,
        'random_state': SEED,
        'class_weight': CLASS_WEIGHT,
        'n_jobs': -1
    }
}