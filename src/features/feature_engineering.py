# src/features/feature_engineering.py
import pandas as pd
import logging
from typing import Dict

from src.config import DATA_DIR, MAIN_TRAIN_FILE, AUX_DATA_FILES, PROCESSED_DATA_DIR, FEATURE_STORE_PATH
from src.features.transformers.auxiliary_aggregator import AuxiliaryFeatureAggregator
from src.features.transformers.feature_creator import FeatureCreator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _load_data_file(filename: str, file_key: str = None) -> pd.DataFrame:
    file_path = DATA_DIR / filename
    logger.info(f"Loading {file_key or filename}")
    df = pd.read_csv(file_path)
    logger.info(f"Shape: {df.shape}")
    return df


def _load_all_data() -> Dict[str, pd.DataFrame]:
    data = {'main': _load_data_file(MAIN_TRAIN_FILE, "Main")}
    for key, fname in AUX_DATA_FILES.items():
        data[key] = _load_data_file(fname, key)
    return data


def create_feature_store(save: bool = True) -> pd.DataFrame:
    logger.info("=" * 60 + "\nCreating Feature Store\n" + "=" * 60)

    data = _load_all_data()
    df_app = data.pop('main')

    # Feature creation
    fc = FeatureCreator(anomaly_value=365243, epsilon=1e-3)
    df_app = fc.fit_transform(df_app)
    logger.info(f"Created {len(fc.created_features_)} features")

    # Auxiliary aggregations
    aggregator = AuxiliaryFeatureAggregator(aux_data=data)
    df_final = aggregator.fit_transform(df_app)
    logger.info(f"Final shape: {df_final.shape}")

    if save:
        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        df_final.to_parquet(FEATURE_STORE_PATH, index=False)
        logger.info(f"Saved to {FEATURE_STORE_PATH}")

    return df_final


if __name__ == '__main__':
    df = create_feature_store()
    print(f"\nFeatures: {len(df.columns)}, Samples: {len(df)}")
    print(f"\nTarget distribution:\n{df['TARGET'].value_counts(normalize=True)}")
