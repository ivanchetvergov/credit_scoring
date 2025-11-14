import pandas as pd
import logging
import argparse
from pathlib import Path
import sys
from typing import List, Dict, Any

from src.models import MODEL_TRAINERS_REGISTRY, AVAILABLE_MODELS, BaseTrainer, SklearnTrainer, PyTorchTrainer
from src.nn_models import PyTorchTrainer
from src.reporting.compare_models import compare_models
from sklearn.model_selection import train_test_split

from src.config import (
    FEATURE_STORE_PATH,
    TARGET_COLUMN,
    ID_COLUMN,
    SEED,
    TEST_SIZE
)
sys.path.append(str(Path(__file__).resolve().parent.parent))

import warnings
warnings.filterwarnings(action='ignore')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def prepare_data(df: pd.DataFrame) -> tuple:
    """
    Подготавливает данные для обучения.
    """
    logger.info("Preparing data for training...")

    # разделение на признаки и таргет
    X = df.drop(columns=[TARGET_COLUMN, ID_COLUMN], axis=1)
    y = df[TARGET_COLUMN]

    logger.info(f"Features shape: {X.shape}")
    logger.info(f"Target shape: {y.shape}")

    # используем Stratify для несбалансированного таргета
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=SEED
    )

    logger.info(f"Train set: {X_train.shape}, positive rate: {y_train.mean():.4f}")
    logger.info(f"Test set: {X_test.shape}, positive rate: {y_test.mean():.4f}")

    return X_train, X_test, y_train, y_test

def train_single_model(
    model_name: str, X_train, y_train,
    X_test, y_test, run_cv: bool = True
) -> Dict[str, Any]:
    """
    Обучает одну модель.
    """
    if model_name not in MODEL_TRAINERS_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}. Available: {AVAILABLE_MODELS}")
    try:
        logger.info(f"{'='*60}")
        logger.info(f"Training model: {model_name}")
        logger.info(f"{'='*60}")

        # 1. получаем класс тренера (CatBoostTrainer, LGBMTrainer, PyTorchTrainer и т.д.)
        TrainerClass = MODEL_TRAINERS_REGISTRY[model_name]

        if TrainerClass in (SklearnTrainer, PyTorchTrainer):
            trainer = TrainerClass(model_name=model_name)
        else:
            trainer = TrainerClass()

        # кросс-валидация (опционально)
        if run_cv:
            trainer.cross_validate(X_train, y_train)

        # обучение и оценка
        final_pipeline, metrics = trainer.train(X_train, y_train, X_test, y_test)

        # сохраняем
        BaseTrainer.save_model(final_pipeline, model_name, metrics)

        return metrics

    except ValueError as e:
        logger.error(f"Validation error for model {model_name}: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Training failed for model {model_name}: {str(e)}", exc_info=True)
        raise


def train_all_models(
    models_to_run: List[str], X_train, y_train,
    X_test, y_test, run_cv: bool = True
) -> Dict[str, Dict[str, Any]]:
    """
    Обучает несколько моделей.
    """
    all_results = {}
    failed_models = []

    for model_name in models_to_run:
        try:
            metrics = train_single_model(
                model_name, X_train, y_train, X_test, y_test, run_cv
            )
            all_results[model_name] = metrics

        except Exception:
            logger.error(f"Model {model_name} failed and will be skipped")
            failed_models.append(model_name)

    if failed_models:
        logger.warning(f"Failed models: {failed_models}")

    return all_results


def train():
    """
    Главный скрипт для запуска обучения.
    Позволяет выбрать модели для запуска через аргумент командной строки.
    """
    parser = argparse.ArgumentParser(
        description="Run the baseline ML training pipeline"
    )
    parser.add_argument(
        'models',
        nargs='*',
        default=['logistic_regression', 'random_forest', 'sgd_classifier', 'lightgbm', 'xgboost', 'catboost'],
        help=f'List of models to train. Available: {", ".join(AVAILABLE_MODELS)}'
    )
    parser.add_argument(
        '--no-cv',
        action='store_true',
        help='Skip cross-validation'
    )
    parser.add_argument(
        '--no-compare',
        action='store_true',
        help='Skip model comparison'
    )
    args = parser.parse_args()

    models_to_run = args.models
    run_cv = not args.no_cv
    run_compare = not args.no_compare

    logger.info("=" * 80)
    logger.info(f"STARTING TRAINING PIPELINE")
    logger.info("=" * 80)
    logger.info(f"Models to train: {', '.join(models_to_run)}")
    logger.info(f"Cross-validation: {'enabled' if run_cv else 'disabled'}")
    logger.info(f"Model comparison: {'enabled' if run_compare else 'disabled'}")
    logger.info("=" * 80)

    # --- 1. загрузка данных ---
    try:
        logger.info(f"Loading feature store from: {FEATURE_STORE_PATH}")
        df = pd.read_parquet(FEATURE_STORE_PATH)
        logger.info(f"Feature store loaded. Shape: {df.shape}")
    except FileNotFoundError:
        logger.error(
            f"Feature store not found at {FEATURE_STORE_PATH}. "
            f"Please run feature_engineering.py first."
        )
        return
    except Exception as e:
        logger.error(f"Error loading feature store: {e}", exc_info=True)
        return

    # --- 2. подготовка данных ---
    try:
        X_train, X_test, y_train, y_test = prepare_data(df)
    except Exception as e:
        logger.error(f"Error preparing data: {e}", exc_info=True)
        return

    # --- 3. обучение моделей ---
    all_results = train_all_models(
        models_to_run, X_train, y_train, X_test, y_test, run_cv
    )

    # --- 4. сравнение моделей ---
    if all_results and run_compare:
        try:
            compare_models(all_results)
        except Exception as e:
            logger.error(f"Error comparing models: {e}", exc_info=True)
    elif not all_results:
        logger.warning("No models were successfully trained for comparison.")

    logger.info("=" * 80)
    logger.info("TRAINING PIPELINE COMPLETED")
    logger.info("=" * 80)


if __name__ == "__main__":
    train()