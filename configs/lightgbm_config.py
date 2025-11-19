# configs/lightgbm_config
from src.config import SEED, SCALE_POS_WEIGHT

MODEL_PARAMS = {
    'lightgbm': {
        'random_state': SEED,
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',

        # learning schedule: slower lr + more rounds (use early stopping in training)
        'n_estimators': 5000,
        'learning_rate': 0.008,   # более медленное обучение
        'n_jobs': -1,

        # complexity control
        'num_leaves': 12,
        'max_depth': 4,           # уменьшенная глубина
        'min_child_samples': 60,
        'min_data_in_leaf': 60,
        'min_gain_to_split': 0.2,

        # stronger regularization
        'reg_alpha': 2.0,         # усиленная регуляризация
        'reg_lambda': 5.0,        # усиленная регуляризация

        # feature / row sampling
        'colsample_bytree': 0.5,
        'subsample': 0.6,
        'subsample_freq': 5,

        # misc
        'verbose': -1,

        # handle class imbalance (adjust to true ratio if available)
        'scale_pos_weight': SCALE_POS_WEIGHT
    },
}