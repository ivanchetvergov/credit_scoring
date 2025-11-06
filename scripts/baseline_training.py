import pandas as pd
import logging
from sklearn.model_selection import train_test_split
from pathlib import Path
import sys
import argparse

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import (
    FEATURE_STORE_PATH,
    TARGET_COLUMN,
    ID_COLUMN,
    SEED,
    TEST_SIZE
)
from src.models.baseline_trainer import BaselineTrainer
from src.reporting.compare_models import compare_models

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def train():
    """
    Главный скрипт для запуска обучения
    Позволяет выбрать модель для запуска через аргумент командной строки
    """
    parser = argparse.ArgumentParser(
        description= "Run the baseline ML training baseline"
    )
    parser.add_argument(
        'models',
        nargs='*',
        default=['logistic_regression', 'random_forest'],
        help='List of models to train (e.g., logistic_regression random_forest).'
    )
    args = parser.parse_args()

    models_to_run = args.models

    logger.info("=" * 60)
    logger.info(f"starting training pipeline for models: {', '.join(models_to_run)}")
    logger.info("=" * 60)

    # 1 загрузка данных
    try:
        df = pd.read_parquet(FEATURE_STORE_PATH)
        logger.info(f"feature store loaded. shape: {df.shape}")
    except FileNotFoundError:
        logger.info(f"feature store was not found at {FEATURE_STORE_PATH}. run feature_engineering first.")
        return

    # 2 разделение данных
    X = df.drop(columns=[TARGET_COLUMN, ID_COLUMN], axis=1)
    y = df[TARGET_COLUMN]

    # используем Stratify для несбаланс таргета
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED
    )
    logger.info(f"data split: train {X_train.shape}, test {X_test.shape}")

    # 3 цикл обучения и сбора результатов
    all_results = {}

    for model_name in models_to_run:
        try:
            trainer = BaselineTrainer(model_name=model_name)

            # кросс валидация
            trainer.cross_validate(X_train, y_train)

            # обучение и оценка
            final_pipeline, metrics = trainer.train(X_train, y_train, X_test, y_test)

            # сохраняем
            BaselineTrainer.save_model(final_pipeline, model_name, metrics)

            all_results[model_name] = metrics

        except ValueError as e:
            logger.error(f"Skipping model {model_name} failed. {str(e)}")
        except Exception as e:
            logger.error(f"Model {model_name} failed. {str(e)}")

    # 4 сравнение
    if all_results:
        compare_models(all_results)
    else:
        logger.warning(f"No models were sucessfuly trained for compare.")

if __name__ == "__main__":
    train()