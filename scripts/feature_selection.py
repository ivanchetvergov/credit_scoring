import json
import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings(action='ignore')

from src.features.data_helpers import get_feature_names
from src.config import SEED, SAVED_MODELS_DIR, RESULTS_DIR, TEST_SIZE

import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

MODEL_NAME = 'lightgbm'
IMPORTANCE_THRESHOLD = 0.0004


def load_full_data():
    """Загрузка полного датасета и разделение на X, y."""
    df = pd.read_parquet(Path('data/processed/feature_store.parquet'))
    X = df.drop(columns=['SK_ID_CURR', 'TARGET'])
    y = df['TARGET']
    return X, y


def load_pipeline():
    """Загрузка обученного LightGBM пайплайна."""
    pipeline = joblib.load(Path(f'{SAVED_MODELS_DIR}/{MODEL_NAME}_pipeline.joblib'))
    return pipeline


def run_feature_selection(model_pipeline, X_data, y_data, threshold: float):
    logger.info(f"====== Starting Permutation Importance ======")

    # 1. разделение на X_test (для надежной оценки важности)
    _, X_test, _, y_test = train_test_split(
        X_data, y_data, test_size=TEST_SIZE, random_state=SEED, stratify=y_data
    )

    # 2. извлечение компонентов
    # 'preprocessor' - Pipeline с DataFrameCoercer, FeatureCreator, ColumnTransformer
    full_preprocessor_pipeline = model_pipeline.named_steps['preprocessor']

    column_transformer = full_preprocessor_pipeline.named_steps['preprocessor']

    model = model_pipeline.named_steps['model']

    # 3. трансформация данных для подачи в PermutationImportance
    X_test_transformed = full_preprocessor_pipeline.transform(X_test)

    transformed_feature_names = get_feature_names(column_transformer)

    # 4. вычисление Permutation Importance
    from eli5.sklearn import PermutationImportance

    logger.info("Calculating importance features...")

    perm = PermutationImportance(
        model,
        random_state=SEED,
        scoring='roc_auc',
        cv='prefit'
    ).fit(X_test_transformed, y_test)

    # 5. сохранение результатов
    importances_df = pd.DataFrame({
        'feature': transformed_feature_names,
        'importance': perm.feature_importances_
    }).sort_values(by='importance', ascending=False)

    # 6. отбор признаков по порогу
    selected_features = importances_df[importances_df['importance'] > threshold]['feature'].tolist()

    features_to_keep = list(set(selected_features))

    logger.info(f" ====== Result feature selection =====")
    logger.info(f"Count features before selection: {X_test_transformed.shape[1]}")
    logger.info(f"Selected features (threshold {threshold}): {len(features_to_keep)}")

    # 7. сохранение нового Feature Contract
    Path(RESULTS_DIR).mkdir(exist_ok=True)  # Использование RESULTS_DIR
    output_path = Path(f'{RESULTS_DIR}') / "feature_order.json"
    with open(output_path, 'w') as f:
        json.dump(features_to_keep, f, indent=4)

    logger.info(f"New Feature Contract saved in {output_path}")
    logger.info(f"Top-5 features: {features_to_keep[:5]}")

    return features_to_keep


if __name__ == '__main__':
    X_data, y_data = load_full_data()
    model_pipeline = load_pipeline()

    run_feature_selection(model_pipeline, X_data, y_data, IMPORTANCE_THRESHOLD)