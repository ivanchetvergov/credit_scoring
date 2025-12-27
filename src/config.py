from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# Пути
DATA_DIR = BASE_DIR / 'data' / "raw"
PROCESSED_DATA_DIR = BASE_DIR / 'data' / "processed"
SAVED_MODELS_DIR = BASE_DIR / 'saved_models'
RESULTS_DIR = BASE_DIR / 'results'

FEATURE_STORE_PATH = PROCESSED_DATA_DIR / "feature_store.parquet"

# Файлы данных
MAIN_TRAIN_FILE = "application_train.csv"
AUX_DATA_FILES = {
    'bureau': 'bureau.csv',
    'bureau_balance': 'bureau_balance.csv',
    'previous_application': 'previous_application.csv'
}

# Целевая переменная и ID
TARGET_COLUMN = "TARGET"
ID_COLUMN = "SK_ID_CURR"

# Фичи (импорт из единого файла)
from configs.raw_features_config import (
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    BIN_CATEGORICAL_FEATURES,
)

# Константы обучения
SEED = 42
TEST_SIZE = 0.25
CV_FOLDS = 5
SCALE_POS_WEIGHT = 15
MAX_CATEGORIES_FOR_OHE = 50

# =============================================================================
# ПАРАМЕТРЫ МОДЕЛЕЙ (объединены из configs/)
# =============================================================================

MODEL_PARAMS = {
    # CatBoost
    'catboost': {
        'random_state': SEED,
        'loss_function': 'Logloss',
        'eval_metric': 'AUC',
        'scale_pos_weight': SCALE_POS_WEIGHT,
        'learning_rate': 0.02,
        'depth': 6,
        'l2_leaf_reg': 10.0,
        'n_estimators': 3000,
        'early_stopping_rounds': 300,
        'verbose': 100,
    },

    # LightGBM
    'lightgbm': {
        'random_state': SEED,
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'n_estimators': 5000,
        'learning_rate': 0.008,
        'n_jobs': -1,
        'num_leaves': 12,
        'max_depth': 4,
        'min_child_samples': 60,
        'min_data_in_leaf': 60,
        'min_gain_to_split': 0.2,
        'reg_alpha': 2.0,
        'reg_lambda': 5.0,
        'colsample_bytree': 0.5,
        'subsample': 0.6,
        'subsample_freq': 5,
        'verbose': -1,
        'scale_pos_weight': SCALE_POS_WEIGHT,
    },

    # XGBoost
    'xgboost': {
        'random_state': SEED,
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'lambda': 3.0,
        'alpha': 2.0,
        'learning_rate': 0.01,
        'max_depth': 4,
        'gamma': 0.3,
        'subsample': 0.6,
        'colsample_bytree': 0.5,
        'n_estimators': 3000,
        'use_label_encoder': False,
        'enable_categorical': True,
        'n_jobs': -1,
        'scale_pos_weight': SCALE_POS_WEIGHT,
    },

    # Sklearn models
    'logistic_regression': {
        'penalty': 'l2',
        'C': 0.5,
        'solver': 'saga',
        'max_iter': 1000,
        'random_state': SEED,
    },
    'random_forest': {
        'n_estimators': 420,
        'max_depth': 5,
        'min_samples_split': 200,
        'min_samples_leaf': 80,
        'random_state': SEED,
        'n_jobs': -1,
        'max_features': 'sqrt',
        'bootstrap': True,
    },
    'sgd_classifier': {
        'loss': 'modified_huber',
        'penalty': 'l2',
        'alpha': 0.0002,
        'max_iter': 1000,
        'random_state': SEED,
        'n_jobs': -1,
    },

    # PyTorch MLP
    'simple_mlp': {
        'hidden_units': [256, 128, 64],
        'dropout': 0.1,
        'n_epochs': 50,
        'batch_size': 256,
        'learning_rate': 0.001,
        'optimizer': 'Adam',
        'loss_function': 'BCEWithLogitsLoss',
        'pos_weight': SCALE_POS_WEIGHT,
        'random_state': SEED,
        'early_stopping_patience': 10,
    },
}

# Список категориальных фичей для CatBoost
CAT_FEATURES_COLS = CATEGORICAL_FEATURES + BIN_CATEGORICAL_FEATURES
