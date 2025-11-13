# configs/lightgbm_config
from src.config import SEED

MODEL_PARAMS = {
    'lightgbm': {
        'random_state': SEED,
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',

        'n_estimators': 2000,
        'learning_rate': 0.03,
        'n_jobs': -1,

        'num_leaves': 31,
        'min_child_samples': 20,
        'max_depth': 8,

        'reg_alpha': 0.2,
        'reg_lambda': 0.2,
        'colsample_bytree': 0.8,
        'subsample': 0.8,

        # 'early_stopping_rounds' :150,
        # вес положительного класса ~ 11.4
        # 'is_unbalance': True
        'scale_pos_weight': 18.5  # увеличим для лучшего Recall
    },
}