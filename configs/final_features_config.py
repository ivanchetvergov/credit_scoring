# configs/final_features_config.py

# =========================================================================
# Feature Contract: Сокращенный список 39 ключевых признаков (по результатам Permutation Importance)
# Цель: Ускорение инференса и снижение переобучения
# =========================================================================

# Исходные 39 фичей сократились до 35 базовых столбцов
# (т.к. 4 OHE-столбца заменены на 4 базовых столбца)

NUMERICAL_FEATURES = [
    'AMT_ANNUITY',
    'AMT_CREDIT',
    'AMT_GOODS_PRICE',
    'AMT_INCOME_TOTAL',
    'BUREAU_AMT_CREDIT_MAX_OVERDUE_MEAN',
    'BUREAU_AMT_CREDIT_SUM_DEBT_MEAN',
    'BUREAU_AMT_CREDIT_SUM_MAX',
    'BUREAU_AMT_CREDIT_SUM_MEAN',
    'BUREAU_DAYS_CREDIT_ENDDATE_MAX',
    'BUREAU_DAYS_CREDIT_MAX',
    'BUREAU_DAYS_CREDIT_MEAN',
    'BUREAU_DAYS_ENDDATE_FACT_MAX',
    'DAYS_BIRTH',
    'DAYS_EMPLOYED',
    'DAYS_ID_PUBLISH',
    'DEF_30_CNT_SOCIAL_CIRCLE',
    'EXT_SOURCE_1',
    'EXT_SOURCE_2',
    'EXT_SOURCE_3',
    'PREV_AMT_ANNUITY_MEAN',
    'PREV_AMT_APPLICATION_MAX',
    'PREV_AMT_DOWN_PAYMENT_SUM',
    'PREV_CNT_PAYMENT_MAX',
    'PREV_CNT_PAYMENT_MEAN',
    'PREV_CNT_PAYMENT_MIN',
    'PREV_DAYS_DECISION_MAX', # добавлено: DAYS_DECISION_MAX
    'PREV_DAYS_FIRST_DRAWING_MEAN',
    'PREV_DAYS_FIRST_DRAWING_MIN',
    'PREV_DAYS_FIRST_DRAWING_SUM',
    'PREV_DAYS_FIRST_DUE_MIN',
    'PREV_DAYS_LAST_DUE_1ST_VERSION_MAX',
    'PREV_DAYS_LAST_DUE_MAX',
    'PREV_DAYS_LAST_DUE_MEAN',
    'PREV_DAYS_LAST_DUE_SUM',
    'REGION_RATING_CLIENT_W_CITY',
]

CATEGORICAL_FEATURES = [
    # Заменены OHE-фичи на базовые категориальные
    'CODE_GENDER',
    'NAME_CONTRACT_TYPE',
    'NAME_EDUCATION_TYPE',
    'NAME_FAMILY_STATUS',
]

BIN_CATEGORICAL_FEATURES = [
    # Заменен OHE-признак на базовый бинарный
    'FLAG_OWN_CAR',
]
