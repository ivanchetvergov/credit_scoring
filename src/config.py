from pathlib import Path

# ========================== #
# 1 константы путей и файлов #
# ========================== #

BASE_DIR = Path(__file__).resolve().parent.parent

# директории
DATA_DIR = BASE_DIR / 'data' / "raw"
PROCESSED_DATA_DIR = BASE_DIR / 'data' / "processed"
MODELS_DIR = BASE_DIR / 'models'

# основной файл данных
MAIN_TRAIN_FILE = "application_train.csv"
MAIN_TRAIN_PATH = DATA_DIR / MAIN_TRAIN_FILE

# файлы, которые мы будем агрегировать (выбираем самые важные для MVP)
AUX_DATA_FILES = {
    'bureau': 'bureau.csv',
    'bureau_balance': 'bureau_balance.csv',
    'previous_application': 'previous_application.csv'
}

# имя и путь для сохранения объединенной и очищенной таблицы (feature store)
FEATURE_STORE_FILE = "feature_store.parquet"
FEATURE_STORE_PATH = PROCESSED_DATA_DIR / FEATURE_STORE_FILE

# имя файла для сохранения обученного ML-пайплайна
MODEL_PIPELINE_NAME = "credit_default_pipeline.joblib"
MODEL_PIPELINE_PATH = MODELS_DIR / MODEL_PIPELINE_NAME


# ========================= #
#  2 определение признаков  #
# ========================= #

TARGET_COLUMN = "TARGET"
DROP_FEATURES = ["SK_ID_CURR"]

# --- признаки из application_train.csv (Обновленный Core Set) ---
NUMERICAL_FEATURES = [
    # финансовые показатели
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "AMT_GOODS_PRICE",

    # скоринговые показатели (критически важны)
    "EXT_SOURCE_2",
    "EXT_SOURCE_3",
    # EXT_SOURCE_1 тоже важен, но здесь мы используем только те, что
    # будем передавать в API, а EXT_SOURCE_1 имеет ~56% пропусков.

    # временные показатели (будут трансформированы)
    "DAYS_BIRTH",
    "DAYS_EMPLOYED",
    "DAYS_REGISTRATION",
]

CATEGORICAL_FEATURES = [
    # основные демографические
    "NAME_CONTRACT_TYPE",
    "CODE_GENDER",
    "FLAG_OWN_CAR",
    "FLAG_OWN_REALTY",
    "CNT_CHILDREN",

    # социальные/Образование
    "NAME_TYPE_SUITE",
    "NAME_INCOME_TYPE",
    "NAME_EDUCATION_TYPE",
    "NAME_FAMILY_STATUS",

    # география и работа
    "REGION_RATING_CLIENT_W_CITY",  # обрабатываем как категориальный
    "ORGANIZATION_TYPE",
]

BIN_CATEGORICAL_FEATURES = [
    "LIVE_REGION_NOT_WORK_REGION",
    "FLAG_DOCUMENT_3",
]

# --- Признаки, которые будут созданы агрегацией из bureau.csv  ---
BUREAU_AGG_FEATURES = [
    "BUREAU_CREDIT_SUM_MEAN",
    "BUREAU_CREDIT_COUNT",
    "BUREAU_DAYS_ENDDATE_MIN",
]

FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES + BIN_CATEGORICAL_FEATURES + BUREAU_AGG_FEATURES


# =============================== #
 # 3 настройка модели и обучения #
# =============================== #

SEED = 42


# ==================== #
 # 4 конфигурация API #
# ==================== #

