import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional

from src.config import RESULTS_DIR

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def compare_models(
    results: Dict[str, Dict[str, Any]],
    save_path: Optional[str] = None,
    sort_by: str = 'Test ROC-AUC'
) -> pd.DataFrame:
    """
    Сравнивает результаты разных моделей.
    :param results: словарь с результатами: {model_name: metrics}
    :param save_path: путь для сохранения результатов
    :param sort_by: метрика для сортировки (default='Test ROC-AUC')
    :return: pd.DataFrame с сравнением моделей
    """
    logger.info("=" * 80)
    logger.info("MODEL COMPARISON")
    logger.info("=" * 80)

    if not results:
        logger.warning("No results to compare")
        return pd.DataFrame()

    try:
        comparison_df = pd.DataFrame({
            model_name: {
                'Train ROC-AUC': metrics['train']['roc_auc'],
                'Test ROC-AUC': metrics['test']['roc_auc'],
                'Test Accuracy': metrics['test']['accuracy'],
                'Test Precision': metrics['test']['precision'],
                'Test Recall': metrics['test']['recall'],
                'Test F1': metrics['test']['f1'],
                'Overfit (Train-Test)': metrics['train']['roc_auc'] - metrics['test']['roc_auc']
            }
            for model_name, metrics in results.items()
        }).T

        if sort_by in comparison_df.columns:
            comparison_df = comparison_df.sort_values(sort_by, ascending=False)

            # выводим результаты
            print("\n" + "=" * 80)
            print("MODEL COMPARISON TABLE")
            print("=" * 80)
            print(comparison_df.round(4).to_string())
            print("=" * 80)

            # определяем лучшую модель
            best_model = comparison_df['Test ROC-AUC'].idxmax()
            best_score = comparison_df.loc[best_model, 'Test ROC-AUC']

            logger.info(f"   Best model: {best_model}")
            logger.info(f"   Test ROC-AUC: {best_score:.4f}")

        if save_path is None:
            save_path = RESULTS_DIR / 'baseline_comparison.csv'
        else:
            save_path = RESULTS_DIR / save_path

        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        comparison_df.to_csv(save_path)
        logger.info(f"\n✓ Comparison saved to: {save_path}")

        return comparison_df

    except Exception as e:
        logger.error(f"Error comparing models: {e}", exc_info=True)
        raise

if __name__ == '__main__':
    # тестирование с фейковыми данными
    test_results = {
        'logistic_regression': {
            'train': {'roc_auc': 0.7656, 'accuracy': 0.6987, 'precision': 0.1678, 'recall': 0.6900, 'f1': 0.2699},
            'test': {'roc_auc': 0.7623, 'accuracy': 0.6987, 'precision': 0.1678, 'recall': 0.6900, 'f1': 0.2699},
            'cv_mean': 0.7600,
            'cv_std': 0.0050,
            'confusion_matrix': [[25000, 3000], [1500, 5000]]
        },
        'random_forest': {
            'train': {'roc_auc': 0.7806, 'accuracy': 0.7117, 'precision': 0.1653, 'recall': 0.6349, 'f1': 0.2623},
            'test': {'roc_auc': 0.7388, 'accuracy': 0.7117, 'precision': 0.1653, 'recall': 0.6349, 'f1': 0.2623},
            'cv_mean': 0.7400,
            'cv_std': 0.0045,
            'confusion_matrix': [[26000, 2000], [2000, 4500]]
        }
    }

    comparison_df = compare_models(test_results)