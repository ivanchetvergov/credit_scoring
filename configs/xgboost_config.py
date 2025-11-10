# configs/xgboost_config.py

from src.config import SEED

MODEL_PARAMS = {
    'xgboost': {
        'random_state': SEED,
        'scale_pos_weight': 20,

        # основные параметры
        'objective': 'binary:logistic',
        'eval_metric': 'auc',

        # регуляризация
        'lambda': 1.0,    # L2 регуляризация
        'alpha': 0.0,     # L1 регуляризация

        # настройки обучения
        'learning_rate': 0.04,
        'max_depth': 6,
        'gamma': 0.2,               # минимальное снижение потерь для разделения
        'subsample': 0.7,           # используем 70% данных для каждого дерева
        'colsample_bytree': 0.7,    # используем 70% фичей для каждого дерева

        # настройки итераций и ранней остановки
        'n_estimators': 3000,
        'early_stopping_rounds': 200,

        # выключаем устаревший кодировщик
        'use_label_encoder': False,
        'n_jobs': -1,
    }
}