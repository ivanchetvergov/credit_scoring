# configs/catboost_config
from src.config import SEED
from configs.raw_features_config import CATEGORICAL_FEATURES, BIN_CATEGORICAL_FEATURES

# CatBoost ожидает список ВСЕХ колонок, содержащих строки.
CAT_FEATURES_COLS = CATEGORICAL_FEATURES + BIN_CATEGORICAL_FEATURES

MODEL_PARAMS = {
    'catboost': {
        'random_state': SEED,
        'loss_function': 'Logloss',
        'eval_metric': 'AUC',
        # улучшенная обработка дисбаланса:
        'scale_pos_weight': 20,  # отношение (негативные / позитивные)

        # настройки обучения
        'learning_rate': 0.04,      # чуть ниже, для лучшего использования early stopping
        'depth': 7,                 # увеличим глубину для большего потенциала
        'l2_leaf_reg': 3.0,

        # настройки итераций и ранней остановки
        'n_estimators': 3000,
        'early_stopping_rounds': 200,   # дадим больше шансов на сходимость
        'verbose': 10,                  # логируем каждые 50 шагов
    }
}