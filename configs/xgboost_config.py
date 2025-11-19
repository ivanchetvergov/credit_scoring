# configs/xgboost_config.py

from src.config import SEED, SCALE_POS_WEIGHT

MODEL_PARAMS = {
    'xgboost': {
        'random_state': SEED,

        # основные параметры
        'objective': 'binary:logistic',
        'eval_metric': 'auc',

        # регуляризация
        'lambda': 3.0,    # усиленная L2 регуляризация
        'alpha': 2.0,     # усиленная L1 регуляризация

        # настройки обучения
        'learning_rate': 0.01,   # более медленное обучение
        'max_depth': 4,          # уменьшенная глубина
        'gamma': 0.3,            # более строгий split
        'subsample': 0.6,        # используем 60% данных для каждого дерева
        'colsample_bytree': 0.5, # используем 50% фичей для каждого дерева

        # настройки итераций и ранней остановки
        'n_estimators': 3000,
        # 'early_stopping_rounds': 200,

        # выключаем устаревший кодировщик
        'use_label_encoder': False,
        'enable_categorical': True,
        'n_jobs': -1,
        'scale_pos_weight': SCALE_POS_WEIGHT,
    }
}