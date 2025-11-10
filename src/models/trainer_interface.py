# src/models/trainer_interface.py
import joblib
import logging
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
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
from typing import Dict, Optional, Tuple, Any
from abc import ABC, abstractmethod

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

class BaseTrainer(ABC):
    """
    Абстрактный класс для обучения и оценки ML-моделей.
    Всю специфичную для модели логику (инициализация, специальные аргументы fit)
    должны реализовывать дочерние классы.
    """

    def __init__(
        self,
        model_name: str,
        model_params: Optional[Dict] = None,
        cv_folds: int = CV_FOLDS
    ) -> None:
        # инициализация параметров модели
        self.model_name = model_name
        self.cv_folds: int = cv_folds
        self.pipeline = None
        self.metrics: Dict[str, Any] = {}

        if model_params is None:
            self.model_params = MODEL_PARAMS.get(model_name, {})
        else:
            self.model_params = model_params

        logger.info(f"Loaded params for {self.model_name}: {self.model_params}")

        # создаем директории
        SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def _get_model(self):
        """
        Абстрактный метод, который должен возвращать конкретную
        инициализированную модель (LGBM, CatBoost, и т.д.).
        """
        pass

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
            y_test: np.ndarray,
            fit_kwargs: Optional[Dict] = None
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

            fit_kwargs = fit_kwargs or {}
            # обучение
            logger.info(f"Fitting model with kwargs: {fit_kwargs}...")
            self.pipeline.fit(X_train, y_train, **fit_kwargs)

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