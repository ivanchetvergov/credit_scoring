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
        'scale_pos_weight': 18.5,  # отношение (негативные / позитивные)

        # настройки обучения
        'learning_rate': 0.035,      # чуть ниже, для лучшего использования early stopping
        'depth': 7,                 # увеличим глубину для большего потенциала
        'l2_leaf_reg': 7.0,

        # настройки итераций и ранней остановки
        'n_estimators': 3000,
        'early_stopping_rounds': 300,    # дадим больше шансов на сходимость
        'verbose': 100,                  # логируем каждые 50 шагов
    }
}