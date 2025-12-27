import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional

from src.config import RESULTS_DIR

logger = logging.getLogger(__name__)


def compare_models(
    results: Dict[str, Dict[str, Any]],
    save_path: Optional[str] = None,
    sort_by: str = 'Test ROC-AUC'
) -> pd.DataFrame:
    """Сравнивает результаты разных моделей и сохраняет в CSV."""
    logger.info("=" * 60 + "\nMODEL COMPARISON\n" + "=" * 60)

    if not results:
        logger.warning("No results to compare")
        return pd.DataFrame()

    comparison_df = pd.DataFrame({
        model_name: {
            'Train ROC-AUC': metrics['train']['roc_auc'],
            'Test ROC-AUC': metrics['test']['roc_auc'],
            'Test Accuracy': metrics['test']['accuracy'],
            'Test Precision': metrics['test']['precision'],
            'Test Recall': metrics['test']['recall'],
            'Test F1': metrics['test']['f1'],
            'CV Mean': metrics.get('cv_mean', np.nan),
            'CV Std': metrics.get('cv_std', np.nan),
            'Overfit (Train-Test)': metrics['train']['roc_auc'] - metrics['test']['roc_auc']
        }
        for model_name, metrics in results.items()
    }).T

    if sort_by in comparison_df.columns:
        comparison_df = comparison_df.sort_values(sort_by, ascending=False)

    print("\n" + "=" * 60 + "\nMODEL COMPARISON TABLE\n" + "=" * 60)
    print(comparison_df.round(4).to_string())
    print("=" * 60)

    best_model = comparison_df['Test ROC-AUC'].idxmax()
    logger.info(f"Best model: {best_model} (ROC-AUC: {comparison_df.loc[best_model, 'Test ROC-AUC']:.4f})")

    save_path = RESULTS_DIR / (save_path or 'baseline_comparison.csv')
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    comparison_df.to_csv(save_path)
    logger.info(f"Saved to: {save_path}")

    return comparison_df
