import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import train_test_split

from src.config import (
    FEATURE_STORE_PATH,
    TARGET_COLUMN,
    BASE_FEATURES,
    RESULTS_DIR,
    SEED,
    TEST_SIZE
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def compare_models(results: dict):
    """
    Сравнивает результаты разных моделей.
    """
    logger.info("=" * 80)
    logger.info("MODEL COMPARISON")
    logger.info("=" * 80)

    comparison_df = pd.DataFrame({
        model_name: {
            'Train ROC-AUC': metrics['train']['roc_auc'],
            'Test ROC-AUC': metrics['test']['roc_auc'],
            'Test Accuracy': metrics['test']['accuracy'],
            'Test Precision': metrics['test']['precision'],
            'Test Recall': metrics['test']['recall'],
            'Test F1': metrics['test']['f1'],
            'CV Mean': metrics.get('cv_mean', np.nan),
            'CV Std': metrics.get('cv_std', np.nan)
        }
        for model_name, metrics in results.items()
    }).T

    print("\n", comparison_df.round(4))

    # сохраняем сравнение
    comparison_df.to_csv(RESULTS_DIR / 'baseline_comparison.csv')
    logger.info(f"\nComparison saved to: {RESULTS_DIR / 'baseline_comparison.csv'}")

    # определяем лучшую модель
    best_model = comparison_df['Test ROC-AUC'].idxmax()
    logger.info(f"\n Best model: {best_model} (Test ROC-AUC: {comparison_df.loc[best_model, 'Test ROC-AUC']:.4f})")

