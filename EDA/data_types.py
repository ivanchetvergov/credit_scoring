import pandas as pd
import numpy as np
import io

from src.config import FEATURE_STORE_PATH

# 1. Загрузите Feature Store
df_feature_store = pd.read_parquet(FEATURE_STORE_PATH)

# Отделяем таргет, чтобы получить X_train
X_train = df_feature_store.drop(columns=['TARGET', 'SK_ID_CURR'])
print(f"Загружен X_train с формой: {X_train.shape}")
print("=====================================================")

# 2. Вывод info() для обзора типов данных
print("--- 1. Общая информация (тип данных и пропуски) ---")
# Используем io.StringIO для перехвата вывода info()
buffer = io.StringIO()
X_train.info(buf=buffer, verbose=False, memory_usage=False)
s_info = buffer.getvalue()
print(s_info)

print("=====================================================")

# 3. Обзор строковых/категориальных колонок (их уникальность и пропуски)
print("--- 2. Обзор нечисловых колонок (строки и категории) ---")
object_df = X_train.select_dtypes(include=['object', 'category'])

if object_df.empty:
    print("В Feature Store нет строковых/категориальных колонок.")
else:
    # Показываем основные статистики для нечисловых фичей (top, freq)
    print(object_df.describe(include='all').T)

print("=====================================================")

# 4. Проверка колонки, которая дала ошибку (NAME_TYPE_SUITE)
print("\n--- 3. Проверка 'NAME_TYPE_SUITE' ---")
if 'NAME_TYPE_SUITE' in X_train.columns:
    print(f"Тип: {X_train['NAME_TYPE_SUITE'].dtype}")
    print("Уникальные значения (топ-5):")
    print(X_train['NAME_TYPE_SUITE'].value_counts().head())
else:
    print("'NAME_TYPE_SUITE' не найдена в Feature Store.")