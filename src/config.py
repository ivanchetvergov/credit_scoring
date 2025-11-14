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

# базовые не обработанные 338 фичей
from configs.raw_features_config import (
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    BIN_CATEGORICAL_FEATURES,
)

# итоговые 35 фичей (с дальнейшим OHE ~ 50)
# from configs.processeed_features_config import (
#     NUMERICAL_FEATURES,
#     CATEGORICAL_FEATURES,
#     BIN_CATEGORICAL_FEATURES,
# )

# =============================== #
 # 3 настройка модели и обучения #
# =============================== #

SEED = 42
TEST_SIZE = 0.25
CV_FOLDS = 5

# --- ИМПОРТ ПАРАМЕТРОВ ---
from configs.catboost_config import MODEL_PARAMS as CATBOOST_PARAMS
from configs.lightgbm_config import MODEL_PARAMS as LGBM_PARAMS
from configs.sklearn_config import MODEL_PARAMS as SKLEARN_PARAMS
from configs.xgboost_config  import MODEL_PARAMS as XGBoost_PARAMS
from configs.torch_config import MODEL_PARAMS as TORCH_PARAMS


# --- ГЛОБАЛЬНЫЙ КАТАЛОГ ПАРАМЕТРОВ ---
MODEL_PARAMS = {}
MODEL_PARAMS.update(CATBOOST_PARAMS)
MODEL_PARAMS.update(LGBM_PARAMS)
MODEL_PARAMS.update(SKLEARN_PARAMS)
MODEL_PARAMS.update(XGBoost_PARAMS)
MODEL_PARAMS.update(TORCH_PARAMS)

# =================================== #
 #  4 настройки для обработки данных #
# =================================== #

MAX_MISSING_PCT = 90.0
# минимальное количество уникальных значений для категориальных признаков
MIN_UNIQUE_VALUES = 2
# максимальное количество категорий для one-hot encoding
MAX_CATEGORIES_FOR_OHE = 50
