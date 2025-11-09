import joblib
import logging
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
from sklearn.pipeline import Pipeline
from lightgbm import LGBMClassifier
from typing import Dict, Optional, Tuple, Any

from src.config import (
    SAVED_MODELS_DIR,
    RESULTS_DIR,
    SEED,
    CV_FOLDS,
    MODEL_PARAMS
)
from src.features.pipelines import get_preprocessing_pipeline

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BaselineTrainer:
    """
    Класс для обучения и оценки baseline моделей.
    :param model_name str название модели
    :param model_params кастомные параметры моделие (если None, исп. из config)
    :param cv_folds кол-во фолдов для кросс валидации
    """
    AVAILABLE_MODELS = ['logistic_regression',
                        'random_forest',
                        'lightgbm'
    ]

    def __init__(
        self,
        model_name: str,
        model_params: Optional[Dict] = None,
        cv_folds: int = CV_FOLDS
    ) -> None:
        if model_name not in self.AVAILABLE_MODELS:
            raise ValueError(
                f"Unknown model: {model_name}. "
                f"Available models: {self.AVAILABLE_MODELS}"
            )

        self.model_name = model_name
        self.model_params = model_params or MODEL_PARAMS.get(model_name, {})
        self.cv_folds = cv_folds
        self.pipeline = None
        self.metrics = {}

        # создаем директории
        SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)


    def _get_model(self):
        """
        Возращает модель по имени.
        :return: sklearn-compatible model Инициализированная модель
        """
        models = {
            'logistic_regression': LogisticRegression,
            'random_forest': RandomForestClassifier,
            'lightgbm': LGBMClassifier
        }

        model_class = models[self.model_name]
        return model_class(**self.model_params)

    @staticmethod
    def _calculate_metrics(
            y_true: np.ndarray,
            y_pred: np.ndarray,
            y_pred_proba: np.ndarray
    ) -> Dict[str, float]:
        """Вычисляет метрики качества модели"""
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            "recall": recall_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred),
            'roc_auc': roc_auc_score(y_true, y_pred_proba)
        }

    def train(
            self,
            X_train: np.ndarray,
            y_train: np.ndarray,
            X_test: np.ndarray,
            y_test: np.ndarray
    ) -> Tuple[Pipeline, Dict[str, Any]]:
        """
        Обучает модель и оценивает на тестовой выборке
        """
        logger.info('=' * 80)
        logger.info(f"Training {self.model_name} model...")
        logger.info('=' * 80)

        try:
            # создаем пайплайн
            preprocessor = get_preprocessing_pipeline()
            model = self._get_model()

            self.pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('model', model)
            ])

            # обучение
            logger.info("Fitting model...")
            self.pipeline.fit(X_train, y_train)

            # предсказание на train
            logger.info("Evaluating on train set...")
            y_train_pred = self.pipeline.predict(X_train)
            y_train_probas = self.pipeline.predict_proba(X_train)[:, 1]
            train_metrics = self._calculate_metrics(y_train, y_train_pred, y_train_probas)

            # предсказание на test
            logger.info("Evaluating on test set...")
            y_test_pred = self.pipeline.predict(X_test)
            y_test_probas = self.pipeline.predict_proba(X_test)[:, 1]
            test_metrics = self._calculate_metrics(y_test, y_test_pred, y_test_probas)

            # сохраняем метрики
            self.metrics.update({
                'train': train_metrics,
                'test': test_metrics,
                'confusion_matrix': confusion_matrix(y_test, y_test_pred).tolist()
            })

            # логируем результаты
            self._log_results()

            # classification report
            logger.info("\nClassification Report (Test):")
            logger.info("\n" + classification_report(y_test, y_test_pred))

            return self.pipeline, self.metrics

        except Exception as e:
            logger.error(f"Error during training: {e}", exc_info=True)
            raise

    def cross_validate(self, X, y):
        """Выполняет кросс-валидацию"""
        logger.info(f"Performing {CV_FOLDS}-fold cross-validation...")

        try:
            cv = StratifiedKFold(
                n_splits=self.cv_folds,
                shuffle=True,
                random_state=SEED
            )

            # создаем пайплайн
            preprocessor = get_preprocessing_pipeline()
            model = self._get_model()
            pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('model', model)
            ])

            # кросс-валидация по ROC-AUC
            cv_scores = cross_val_score(
                pipeline, X, y,
                cv=cv,
                scoring='roc_auc',
                n_jobs=-1,
                error_score='raise'
            )

            logger.info(f"Cross-validation ROC-AUC scores: {cv_scores}")
            logger.info(f"Mean CV ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

            self.metrics['cv_scores'] = cv_scores.tolist()
            self.metrics['cv_mean'] = float(cv_scores.mean())
            self.metrics['cv_std'] = float(cv_scores.std())

            return cv_scores

        except Exception as e:
            logger.error(f"Error during cross-validation: {e}", exc_info=True)
            raise

    def _log_results(self):
        """Логирует результаты обучения."""
        logger.info("=" * 80)
        logger.info(f"RESULTS FOR {self.model_name.upper()}")
        logger.info("=" * 80)

        logger.info("Train Metrics:")
        for metric, value in self.metrics['train'].items():
            logger.info(f"  {metric}: {value:.4f}")

        logger.info("Test Metrics:")
        for metric, value in self.metrics['test'].items():
            logger.info(f"  {metric}: {value:.4f}")

        # проверка на overfitting
        train_auc = self.metrics['train']['roc_auc']
        test_auc = self.metrics['test']['roc_auc']
        overfit = train_auc - test_auc

        logger.info(f"Overfitting check:")
        logger.info(f"  Train ROC-AUC: {train_auc:.4f}")
        logger.info(f"  Test ROC-AUC: {test_auc:.4f}")
        logger.info(f"  Difference: {overfit:.4f}")

        if overfit > 0.05:
            logger.warning("  Model might be overfitting!")
        else:
            logger.info("  Model generalization looks good")

        if 'confusion_matrix' in self.metrics:
            cm = np.array(self.metrics['confusion_matrix'])
            logger.info(f"\nConfusion Matrix:")
            logger.info(f"  TN={cm[0, 0]}, FP={cm[0, 1]}")
            logger.info(f"  FN={cm[1, 0]}, TP={cm[1, 1]}")

    @staticmethod
    def save_model(pipeline, model_name: str, metrics: dict):
        """Сохраянет обученую модель"""
        try:
            # убедимся, что директории существуют
            SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
            RESULTS_DIR.mkdir(parents=True, exist_ok=True)

            # сохраняем модель
            model_path = SAVED_MODELS_DIR / f"{model_name}_pipeline.joblib"
            joblib.dump(pipeline, model_path)
            logger.info(f"  Saved model pipeline to {model_path}")

            # сохраняем метрики
            metrics_path = RESULTS_DIR / f"{model_name}_metrics.joblib"
            joblib.dump(metrics, metrics_path)
            logger.info(f"  Saved metrics to {metrics_path}")

            return model_path, metrics_path

        except Exception as e:
            logger.error(f"Error saving model: {e}", exc_info=True)
            raise

    @staticmethod
    def load_model(model_name: str) -> Tuple[Pipeline, Dict[str, Any]]:
        """
        Загружает сохраненную модель и метрики.
        """
        try:
            model_path = SAVED_MODELS_DIR / f"{model_name}_pipeline.joblib"
            metrics_path = RESULTS_DIR / f"{model_name}_metrics.joblib"

            if not model_path.exists():
                raise FileNotFoundError(f"Model not found: {model_path}")

            pipeline = joblib.load(model_path)
            logger.info(f"✓ Loaded model from {model_path}")

            metrics = {}
            if metrics_path.exists():
                metrics = joblib.load(metrics_path)
                logger.info(f"✓ Loaded metrics from {metrics_path}")
            else:
                logger.warning(f"Metrics file not found: {metrics_path}")

            return pipeline, metrics

        except Exception as e:
            logger.error(f"Error loading model: {e}", exc_info=True)
            raise