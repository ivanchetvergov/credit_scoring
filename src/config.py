from pathlib import Path

# ========================== #
# 1 константы путей и файлов #
# ========================== #

# определяем базовую директорию проекта
BASE_DIR = Path(__file__).resolve().parent.parent

# директории для хранения данных, моделей и результатов
DATA_DIR = BASE_DIR / 'data' / "raw"
PROCESSED_DATA_DIR = BASE_DIR / 'data' / "processed"
SAVED_MODELS_DIR = BASE_DIR / 'saved_models'
RESULTS_DIR = BASE_DIR / 'results'

# основной файл данных
MAIN_TRAIN_FILE = "application_train.csv"

# вспомогательные файлы, используемые для агрегации
AUX_DATA_FILES = {
    'bureau': 'bureau.csv',
    'bureau_balance': 'bureau_balance.csv',
    'previous_application': 'previous_application.csv'
}

# имя и путь для feature store (parquet - быстро и надежно)
FEATURE_STORE_FILE = "feature_store.parquet"
FEATURE_STORE_PATH = PROCESSED_DATA_DIR / FEATURE_STORE_FILE

# имя файла для сохранения обученного ml-пайплайна
MODEL_PIPELINE_NAME = "credit_default_pipeline.joblib"
MODEL_PATH = SAVED_MODELS_DIR / MODEL_PIPELINE_NAME


# ========================= #
#  2 определение признаков  #
# ========================= #

TARGET_COLUMN = "TARGET"
ID_COLUMN = "SK_ID_CURR"

from configs.features_config import (
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    BIN_CATEGORICAL_FEATURES,
)

# =============================== #
 # 3 настройка модели и обучения #
# =============================== #

SEED = 42
TEST_SIZE = 0.25
CV_FOLDS = 5

# параметры для разных моделей
MODEL_PARAMS = {
    'logistic_regression': {
        'max_iter': 1000,
        'random_state': SEED,
        'class_weight': 'balanced'
    },
    'random_forest': {
        'n_estimators': 250,
        'max_depth': 8,
        'min_samples_split': 60,
        'min_samples_leaf': 20,
        'random_state': SEED,
        'class_weight': 'balanced',
        'n_jobs': -1,
        'verbose': 0
    },
    'lightgbm': {
        'random_state': SEED,
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'n_estimators': 1000,
        'learning_rate': 0.05,
        'n_jobs': -1,
        'num_leaves': 15,
        'min_child_samples': 70,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'colsample_bytree': 0.8,
        'subsample': 0.8,
        'max_depth': 6
    }
}

# =================================== #
 #  4 настройки для обработки данных #
# =================================== #

MAX_MISSING_PCT = 90.0
# минимальное количество уникальных значений для категориальных признаков
MIN_UNIQUE_VALUES = 2
# максимальное количество категорий для one-hot encoding
MAX_CATEGORIES_FOR_OHE = 50
