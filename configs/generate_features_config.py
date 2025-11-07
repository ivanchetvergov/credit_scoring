# configs/generate_config.py
import pandas as pd
from pathlib import Path

from src.config import FEATURE_STORE_PATH

# Предполагаемый путь к Feature Store
FEATURE_STORE_PATH = Path(FEATURE_STORE_PATH)


def generate_feature_lists(df: pd.DataFrame):
    """Генерирует списки признаков на основе их dtypes."""

    # Исключаемые колонки
    EXCLUDED_COLS = ['SK_ID_CURR', 'TARGET']

    # 1. Отделяем признаки для модели
    X_data = df.drop(columns=EXCLUDED_COLS, errors='ignore')

    # 2. Определяем типы
    numerical_cols = X_data.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = X_data.select_dtypes(include=['object']).columns.tolist()

    # 3. Дополнительная фильтрация: Бинарные
    # Определяем бинарные колонки. Включаем только исходные,
    # а не агрегации, которые могут быть 0.0 или 1.0 (float)

    # Сначала используем список исходных бинарных фичей (если он у вас есть)
    # Если нет, оставим их в numerical, а затем вручную переместим.

    # Для простоты и избежания NaN/Float:
    # Ищем фичи с небольшим количеством уникальных значений
    bin_categorical_cols = []

    # Пройдемся по числовым, ища бинарные (0, 1)
    temp_numerical = numerical_cols.copy()
    for col in numerical_cols:
        # Проверка: только два уникальных значения (0 и 1), без учета NaN
        if X_data[col].nunique(dropna=True) == 2:
            # Если это исходная бинарная фича (не OHE и не агрегация),
            # перемещаем в BIN_CATEGORICAL_FEATURES
            if X_data[col].min() in [0, 1] and X_data[col].max() in [0, 1]:
                bin_categorical_cols.append(col)
                temp_numerical.remove(col)

    # 4. Финальные списки
    numerical_cols = temp_numerical

    print("=" * 80)
    print("ГОТОВЫЕ СПИСКИ ДЛЯ src/config.py")
    print("=" * 80)
    print(f"\n# Всего фичей для модели: {len(numerical_cols) + len(categorical_cols) + len(bin_categorical_cols)}")
    print(f"# Обнаружено числовых: {len(numerical_cols)}")
    print(f"# Обнаружено категориальных (строковых): {len(categorical_cols)}")
    print(f"# Обнаружено бинарных (0/1): {len(bin_categorical_cols)}")

    print("\nNUMERICAL_FEATURES = [")
    for col in sorted(numerical_cols):
        print(f"    '{col}',")
    print("]")

    print("\nCATEGORICAL_FEATURES = [")
    for col in sorted(categorical_cols):
        print(f"    '{col}',")
    print("]")

    print("\nBIN_CATEGORICAL_FEATURES = [")
    for col in sorted(bin_categorical_cols):
        print(f"    '{col}',")
    print("]")


if __name__ == '__main__':
    try:
        df = pd.read_parquet(FEATURE_STORE_PATH)
        generate_feature_lists(df)
    except FileNotFoundError:
        print(f"Ошибка: Файл не найден по пути {FEATURE_STORE_PATH}")