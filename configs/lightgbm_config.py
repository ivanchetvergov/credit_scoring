# configs/lightgbm_config
from src.config import SEED

MODEL_PARAMS = {
    'lightgbm': {
        'random_state': SEED,
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'n_estimators': 500,
        'learning_rate': 0.05,
        'n_jobs': -1,
        'num_leaves': 10,
        'min_child_samples': 50,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'colsample_bytree': 0.8,
        'subsample': 0.8,
        'max_depth': 5,
        # вес положительного класса ~ 11.4
        # 'is_unbalance': True
        'scale_pos_weight': 20.0  # увеличим для лучшего Recall
    },
}