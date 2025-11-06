from pathlib import Path

# ========================== #
# 1 константы путей и файлов #
# ========================== #

# определяем базовую директорию проекта
BASE_DIR = Path(__file__).resolve().parent.parent

# директории для хранения данных, моделей и результатов
DATA_DIR = BASE_DIR / 'data' / "raw"
PROCESSED_DATA_DIR = BASE_DIR / 'data' / "processed"
MODELS_DIR = BASE_DIR / 'saved_models'
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
MODEL_PATH = MODELS_DIR / MODEL_PIPELINE_NAME


# ========================= #
#  2 определение признаков  #
# ========================= #

TARGET_COLUMN = "TARGET"
ID_COLUMN = "SK_ID_CURR"

# --- признаки из application_train.csv (исходные) ---

# числовые признаки
NUMERICAL_FEATURES = [
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "AMT_GOODS_PRICE",
    "EXT_SOURCE_2",
    "EXT_SOURCE_3",
    "DAYS_BIRTH",
    "DAYS_EMPLOYED",
    "DAYS_REGISTRATION",
    "DAYS_ID_PUBLISH",
    "CNT_FAM_MEMBERS",
    "REGION_POPULATION_RELATIVE",
]

# категориальные признаки (catboost будет работать с ними как с текстом)
CATEGORICAL_FEATURES = [
    "NAME_CONTRACT_TYPE",
    "CODE_GENDER",
    "FLAG_OWN_CAR",
    "FLAG_OWN_REALTY",
    "CNT_CHILDREN",
    "NAME_TYPE_SUITE",
    "NAME_INCOME_TYPE",
    "NAME_EDUCATION_TYPE",
    "NAME_FAMILY_STATUS",
    "NAME_HOUSING_TYPE",
    "REGION_RATING_CLIENT",
    "REGION_RATING_CLIENT_W_CITY",
    "ORGANIZATION_TYPE",
    "OCCUPATION_TYPE",
    "WEEKDAY_APPR_PROCESS_START",
    "HOUR_APPR_PROCESS_START",
    "EMERGENCYSTATE_MODE",
    "FONDKAPREMONT_MODE",
    "HOUSETYPE_MODE",
    "WALLSMATERIAL_MODE",
]

# бинарные признаки
BIN_CATEGORICAL_FEATURES = [
    "LIVE_REGION_NOT_WORK_REGION",
    "REG_CITY_NOT_LIVE_CITY",
    "FLAG_DOCUMENT_3",
    "FLAG_PHONE",
    "FLAG_EMAIL",
    "REG_REGION_NOT_LIVE_REGION",
    "LIVE_CITY_NOT_WORK_CITY",
]

# --- добавляем созданные нами признаки (featurecreator) ---
NUMERICAL_FEATURES.extend([
    "CREDIT_INCOME_RATIO",
    "ANNUITY_INCOME_RATIO",
    "AGE_YEARS",
    "EMPLOYMENT_YEARS",
    "GOODS_PRICE_TO_CREDIT_RATIO",
])

BIN_CATEGORICAL_FEATURES.append("DAYS_EMPLOYED_ANOM")

# --- финальные списки признаков ---

# все признаки, которые мы начинаем обрабатывать (исходные + созданные)
BASE_FEATURES = (
    NUMERICAL_FEATURES +
    CATEGORICAL_FEATURES +
    BIN_CATEGORICAL_FEATURES
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
        'n_estimators': 200,
        'max_depth': 10,
        'min_samples_split': 42,
        'random_state': SEED,
        'class_weight': 'balanced',
        'n_jobs': -1
    }
}

