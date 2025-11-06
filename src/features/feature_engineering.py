import pandas as pd
import numpy as np
import logging

from src.config import (
    DATA_DIR,
    MAIN_TRAIN_FILE,
    AUX_DATA_FILES,
    PROCESSED_DATA_DIR,
    FEATURE_STORE_PATH
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def _aggregate_bureau_balance(df_bb: pd.DataFrame) -> pd.DataFrame:
    """
    Агрегирует ежемесячные статусы просрочки из bureau_balance на уровень
    каждого предыдущего кредита (SK_ID_BUREAU)

    :param df_bb: Bureau Balance DataFrame
    :return: aggregated Bureau Balance DataFrame
    """
    logger.debug("Aggregating Bureau Balance features...")

    # --- 1 считаем частоту каждого статуса просрочки
    df_bb_agg = (
        df_bb
        .groupby('SK_ID_BUREAU')['STATUS']
        .value_counts(normalize=True)
        .unstack(fill_value=0)
    )

    # --- 2 переименовываем колонки
    df_bb_agg.columns = [f'BB_STATUS_PCT_{col}' for col in df_bb_agg.columns]

    # --- 3 добавляем длину кредитной истории
    df_bb_agg['BB_HISTORY_LENGTH'] = df_bb.groupby('SK_ID_BUREAU').size()

    logger.debug(f"Bureau Balance features created. Shape: {df_bb_agg.shape}")
    return df_bb_agg.reset_index()


def _aggregate_bureau(df_bureau: pd.DataFrame, df_bb_agg: pd.DataFrame) -> pd.DataFrame:
    """
    Агрегирует Bureau данные на уровень клиента (SK_ID_CURR)

    :param df_bureau: Bureau DataFrame
    :param df_bb_agg: Aggregated Bureau Balance DataFrame
    :return: Aggregated Bureau features at client level
    """
    logger.debug("Merging Bureau with Bureau Balance...")

    # слияние Bureau с Bureau Balance
    df_bureau = df_bureau.merge(df_bb_agg, on='SK_ID_BUREAU', how='left')

    logger.debug("Aggregating Bureau features to client level (SK_ID_CURR)...")

    # выбираем только числовые колонки для агрегации
    num_cols = df_bureau.select_dtypes(include=np.number).columns.tolist()
    num_cols = [col for col in num_cols if col not in ['SK_ID_CURR', 'SK_ID_BUREAU']]

    # определяем функции агрегации
    agg_funcs = ['mean', 'max', 'sum', 'min']
    agg_dict = {col: agg_funcs for col in num_cols}

    # добавляем подсчет количества кредитов
    agg_dict['SK_ID_BUREAU'] = ['count']

    # агрегация с созданием мультииндекса
    df_bureau_agg = df_bureau.groupby('SK_ID_CURR').agg(agg_dict)

    # формируем названия колонок
    df_bureau_agg.columns = [
        f'BUREAU_{col[0]}_{col[1].upper()}' if col[0] != 'SK_ID_BUREAU'
        else 'BUREAU_CREDIT_COUNT'
        for col in df_bureau_agg.columns
    ]

    logger.debug(f"Bureau aggregation complete. Created {len(df_bureau_agg.columns)} features")

    return df_bureau_agg.reset_index()


def _aggregate_previous_application(df_prev: pd.DataFrame) -> pd.DataFrame:
    """
    Агрегирует данные о предыдущих заявках на уровень клиента

    :param df_prev: Previous Application DataFrame
    :return: Aggregated Previous Application features
    """
    logger.debug("Aggregating Previous Application features...")

    # выбираем числовые колонки
    num_cols = df_prev.select_dtypes(include=np.number).columns.tolist()
    num_cols = [col for col in num_cols if col not in ['SK_ID_CURR', 'SK_ID_PREV']]

    agg_funcs = ['mean', 'max', 'min', 'sum']
    agg_dict = {col: agg_funcs for col in num_cols}
    agg_dict['SK_ID_PREV'] = ['count']

    df_prev_agg = df_prev.groupby('SK_ID_CURR').agg(agg_dict)

    df_prev_agg.columns = [
        f'PREV_{col[0]}_{col[1].upper()}' if col[0] != 'SK_ID_PREV'
        else 'PREV_APPLICATION_COUNT'
        for col in df_prev_agg.columns
    ]

    logger.debug(f"Previous Application aggregation complete. Created {len(df_prev_agg.columns)} features")

    return df_prev_agg.reset_index()


def create_feature_store() -> pd.DataFrame:
    """
    Главная функция feature engineering'a: загрузка, агрегация, слияние
    и сохранение feature_store

    :return: pd.DataFrame (датафрейм для дальнейшей работы с датасетом)
    """
    logger.info("=" * 80)
    logger.info("Starting Feature Store creation...")
    logger.info("=" * 80)

    # --- 1 загрузка данных ---
    logger.info(f"Loading main data: {MAIN_TRAIN_FILE}")
    df_app = pd.read_csv(DATA_DIR / MAIN_TRAIN_FILE)
    logger.info(f"Main data shape: {df_app.shape}")

    logger.info(f"Loading bureau data: {AUX_DATA_FILES['bureau']}")
    df_bureau = pd.read_csv(DATA_DIR / AUX_DATA_FILES['bureau'])
    logger.info(f"Bureau data shape: {df_bureau.shape}")

    logger.info(f"Loading bureau_balance data: {AUX_DATA_FILES['bureau_balance']}")
    df_bureau_balance = pd.read_csv(DATA_DIR / AUX_DATA_FILES['bureau_balance'])
    logger.info(f"Bureau Balance data shape: {df_bureau_balance.shape}")

    logger.info(f"Loading previous_application data: {AUX_DATA_FILES['previous_application']}")
    df_prev = pd.read_csv(DATA_DIR / AUX_DATA_FILES['previous_application'])
    logger.info(f"Previous Application data shape: {df_prev.shape}")

    # --- 2 агрегация Bureau Balance ---
    df_bb_agg = _aggregate_bureau_balance(df_bureau_balance)

    # --- 3 агрегация Bureau ---
    df_bureau_agg = _aggregate_bureau(df_bureau, df_bb_agg)

    # --- 4 агрегация Previous Application ---
    df_prev_agg = _aggregate_previous_application(df_prev)

    # --- 5 финальное слияние ---
    logger.info("Merging all features with main application data...")
    df_final = df_app.copy()

    df_final = df_final.merge(df_bureau_agg, on='SK_ID_CURR', how='left')
    logger.info(f"After Bureau merge: {df_final.shape}")

    df_final = df_final.merge(df_prev_agg, on='SK_ID_CURR', how='left')
    logger.info(f"After Previous Application merge: {df_final.shape}")

    # --- 6 сохранение Feature Store ---
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    df_final.to_parquet(FEATURE_STORE_PATH, index=False)
    # df_final.to_csv(FEATURE_STORE_PATH, index=False)

    logger.info("=" * 80)
    logger.info(f"Feature store created successfully!")
    logger.info(f"Final shape: {df_final.shape}")
    logger.info(f"Saved to: {FEATURE_STORE_PATH}")
    logger.info("=" * 80)

    return df_final


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