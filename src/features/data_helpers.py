# src/features/data_helpers.py
import pandas as pd
from typing import List
import logging
from sklearn.compose import ColumnTransformer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ========================== #
 #  вспомогательные функции #
# ========================== #

def get_feature_names(preprocessor: ColumnTransformer) -> list:
    """
    Извлекает названия признаков после ColumnTransformer.
    :param preprocessor: Fitted ColumnTransformer
    :return: Список названий признаков после трансформации
    """
    feature_names = []

    for name, transformer, features in preprocessor.transformers_:
        if name == 'remainder':
            continue

        if hasattr(transformer, 'get_feature_names_out'):
            names = transformer.get_feature_names_out(features)
        else:
            names = features

        feature_names.extend(names)

    return feature_names


def analyze_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Анализирует пропущенные значения в датасете.

    :param df: DataFrame для анализа
    :return: DataFrame со статистикой по пропускам
    """
    missing_stats = pd.DataFrame({
        'feature': df.columns,
        'missing_count': df.isnull().sum().values,
        'missing_pct': (df.isnull().sum() / len(df) * 100).values
    })

    missing_stats = missing_stats[missing_stats['missing_count'] > 0].sort_values(
        'missing_pct', ascending=False
    )

    return missing_stats


def validate_features(df: pd.DataFrame, required_features: List[str]) -> bool:
    """
    Проверяет наличие всех необходимых признаков в DataFrame.

    :param df: pd.DataFrame для проверки
    :param required_features: List[str] cписок необходимых признаков
    :return: bool. True если все признаки присутствуют, иначе False
    """
    missing = [f for f in required_features if f not in df.columns]

    if missing:
        logger.error(f"Missing required features: {missing}")
        return False

    logger.info(f"All {len(required_features)} required features are present")
    return True
