# src/features/feature_engineering.py
import pandas as pd
import logging
from typing import Optional, Dict

from src.config import (
    DATA_DIR,
    MAIN_TRAIN_FILE,
    AUX_DATA_FILES,
    PROCESSED_DATA_DIR,
    FEATURE_STORE_PATH
)

from src.features.custom_transformers import AuxiliaryFeatureAggregator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# --- вспомогательные функции для I/O ---

def _load_data_file(filename: str, file_key: Optional[str] = None) -> pd.DataFrame:
    """
    Загружает CSV файл с обработкой ошибок.
    """
    file_path = DATA_DIR / filename
    file_desc = file_key if file_key else filename

    logger.info(f"Loading {file_desc}: {filename}")

    try:
        df = pd.read_csv(file_path)
        logger.info(f"{file_desc} shape: {df.shape}")
        return df

    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise
    except Exception as e:
        logger.error(f"Error loading {file_desc}: {e}", exc_info=True)
        raise


def _load_all_data() -> Dict[str, pd.DataFrame]:
    """
    Загружает все необходимые датафреймы в словарь.
    """
    # 1. загрузка основного файла
    data = {'main': _load_data_file(
        MAIN_TRAIN_FILE, "Main application data"
    )}

    # 2. загрузка вспомогательных файлов
    for key, filename in AUX_DATA_FILES.items():
        data[key] = _load_data_file(filename, key)

    return data


# --- главная функция Feature Store ---

def create_feature_store(save: bool = True) -> pd.DataFrame:
    """
    Главная функция feature engineering'a: загрузка, агрегация, слияние
    и сохранение feature_store.

    ИСПРАВЛЕНО: теперь использует fit_transform вместо просто transform.
    """
    logger.info("=" * 80)
    logger.info("Starting Feature Store creation...")
    logger.info("=" * 80)

    try:
        # --- 1 загрузка данных ---
        data = _load_all_data()
        df_app = data.pop('main')  # Основной датафрейм (база)

        # --- 2 агрегация и слияние (ИСПОЛЬЗУЕМ ТРАНСФОРМЕР) ---
        logger.info("Starting Auxiliary Feature Aggregation...")

        # инициализируем агрегатор с загруженными вспомогательными данными
        aggregator = AuxiliaryFeatureAggregator(aux_data=data)

        # используем fit_transform вместо просто transform
        df_final = aggregator.fit_transform(df_app)

        logger.info(f"Feature Aggregation complete. Final shape: {df_final.shape}")

        # --- 3 сохранение Feature Store ---
        if save:
            PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
            # используем Parquet, как более эффективный формат для Feature Store
            df_final.to_parquet(FEATURE_STORE_PATH, index=False)

            logger.info("=" * 80)
            logger.info(f"Feature store created successfully!")
            logger.info(f"Final shape: {df_final.shape}")
            logger.info(f"Saved to: {FEATURE_STORE_PATH}")
            logger.info("=" * 80)
        else:
            logger.info("Feature store created but not saved (save=False)")

        return df_final

    except Exception as e:
        logger.error(f"Error creating feature store: {e}", exc_info=True)
        raise


if __name__ == '__main__':
    df = create_feature_store()

    # показываем статистику
    print("\n" + "=" * 80)
    print("FEATURE STORE STATISTICS")
    print("=" * 80)
    print(f"\nTotal features: {len(df.columns)}")
    print(f"Total samples: {len(df)}")
    print(f"\nMissing values per feature (top 10):")
    print(df.isnull().sum().sort_values(ascending=False).head(10))
    print(f"\nTarget distribution:")
    print(df['TARGET'].value_counts(normalize=True))