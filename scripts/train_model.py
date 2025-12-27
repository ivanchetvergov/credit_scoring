#!/usr/bin/env python
# scripts/train_model.py
"""CLI для обучения моделей."""
import argparse
import logging
import warnings
import pandas as pd

from src.config import FEATURE_STORE_PATH, TARGET_COLUMN, ID_COLUMN, SEED, TEST_SIZE
from src.models import MODEL_TRAINERS, AVAILABLE_MODELS, BaseTrainer, SklearnTrainer
from src.reporting.compare_models import compare_models
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def prepare_data(df: pd.DataFrame):
    """Разделяет данные на train/test."""
    X = df.drop(columns=[TARGET_COLUMN, ID_COLUMN])
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED
    )

    logger.info(f"Train: {X_train.shape}, Test: {X_test.shape}, Positive rate: {y_train.mean():.4f}")
    return X_train, X_test, y_train, y_test


def train_model(model_name: str, X_train, y_train, X_test, y_test, run_cv: bool = True):
    """Обучает одну модель."""
    if model_name not in MODEL_TRAINERS:
        raise ValueError(f"Unknown model: {model_name}. Available: {AVAILABLE_MODELS}")

    TrainerClass = MODEL_TRAINERS[model_name]

    if TrainerClass == SklearnTrainer:
        trainer = TrainerClass(model_name=model_name)
    else:
        trainer = TrainerClass()

    if run_cv:
        trainer.cross_validate(X_train, y_train)

    pipeline, metrics = trainer.train(X_train, y_train, X_test, y_test)
    BaseTrainer.save_model(pipeline, model_name, metrics)

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train ML models")
    parser.add_argument('models', nargs='*', default=AVAILABLE_MODELS,
                       help=f'Models to train: {", ".join(AVAILABLE_MODELS)}')
    parser.add_argument('--no-cv', action='store_true', help='Skip cross-validation')
    parser.add_argument('--no-compare', action='store_true', help='Skip model comparison')
    args = parser.parse_args()

    logger.info(f"{'='*60}\nSTARTING TRAINING\n{'='*60}")
    logger.info(f"Models: {args.models}, CV: {not args.no_cv}")

    try:
        df = pd.read_parquet(FEATURE_STORE_PATH)
        logger.info(f"Loaded feature store: {df.shape}")
    except FileNotFoundError:
        logger.error(f"Feature store not found: {FEATURE_STORE_PATH}")
        return

    X_train, X_test, y_train, y_test = prepare_data(df)

    results = {}
    for model_name in args.models:
        try:
            results[model_name] = train_model(
                model_name, X_train, y_train, X_test, y_test, run_cv=not args.no_cv
            )
        except Exception as e:
            logger.error(f"Failed: {model_name} - {e}")

    if results and not args.no_compare:
        compare_models(results)

    logger.info(f"{'='*60}\nTRAINING COMPLETED\n{'='*60}")


if __name__ == '__main__':
    main()
