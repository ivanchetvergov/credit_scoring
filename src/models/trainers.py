# src/models/trainers.py
"""Унифицированные trainer-классы для ML моделей."""
import joblib
import logging
import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple, Any

from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score,
    recall_score, f1_score, confusion_matrix, classification_report
)
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

from src.config import SAVED_MODELS_DIR, RESULTS_DIR, SEED, CV_FOLDS, MODEL_PARAMS
from src.pipelines import get_preprocessing_pipeline

logger = logging.getLogger(__name__)


class BaseTrainer(ABC):
    """Базовый класс для обучения ML-моделей."""

    def __init__(self, model_name: str, model_params: Optional[Dict] = None, cv_folds: int = CV_FOLDS):
        self.model_name = model_name
        self.cv_folds = cv_folds
        self.pipeline = None
        self.metrics: Dict[str, Any] = {}
        self.model_params = model_params or MODEL_PARAMS.get(model_name, {})

        SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        logger.info(f"Params for {self.model_name}: {self.model_params}")

    @abstractmethod
    def _get_model(self):
        """Возвращает инициализированную модель."""
        pass

    def _get_preprocessor(self) -> Pipeline:
        return get_preprocessing_pipeline(self.model_name)

    def _build_fit_kwargs(self, X_train, y_train, X_test, y_test) -> Dict:
        """Kwargs для fit() — переопределяется в подклассах для early stopping и т.д."""
        return {}

    @staticmethod
    def _calculate_metrics(y_true, y_pred, y_pred_proba) -> Dict[str, float]:
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_true, y_pred_proba)
        }

    def train(self, X_train, y_train, X_test, y_test) -> Tuple[Pipeline, Dict]:
        """Обучает модель и оценивает на тестовой выборке."""
        logger.info(f"{'='*60}\nTraining {self.model_name}...\n{'='*60}")

        preprocessor = self._get_preprocessor()
        model = self._get_model()
        self.pipeline = Pipeline([('preprocessor', preprocessor), ('model', model)])

        # Логируем количество фичей
        try:
            X_transformed = preprocessor.fit_transform(X_train, y_train)
            n_features = X_transformed.shape[1] if hasattr(X_transformed, 'shape') else 'unknown'
            logger.info(f"Training on {n_features} features")
            preprocessor.fit(X_train, y_train)  # refit для чистого состояния
        except Exception as e:
            logger.debug(f"Could not count features: {e}")

        fit_kwargs = self._build_fit_kwargs(X_train, y_train, X_test, y_test)
        self.pipeline.fit(X_train, y_train, **fit_kwargs)

        # Метрики
        train_metrics = self._calculate_metrics(
            y_train, self.pipeline.predict(X_train), self.pipeline.predict_proba(X_train)[:, 1]
        )
        test_metrics = self._calculate_metrics(
            y_test, self.pipeline.predict(X_test), self.pipeline.predict_proba(X_test)[:, 1]
        )

        self.metrics = {
            'train': train_metrics,
            'test': test_metrics,
            'confusion_matrix': confusion_matrix(y_test, self.pipeline.predict(X_test)).tolist()
        }

        self._log_results()
        logger.info(f"\n{classification_report(y_test, self.pipeline.predict(X_test))}")
        return self.pipeline, self.metrics

    def cross_validate(self, X, y):
        """Выполняет кросс-валидацию."""
        logger.info(f"Running {self.cv_folds}-fold CV...")
        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=SEED)
        pipeline = Pipeline([('preprocessor', self._get_preprocessor()), ('model', self._get_model())])
        scores = cross_val_score(pipeline, X, y, cv=cv, scoring='roc_auc', n_jobs=-1)

        logger.info(f"CV ROC-AUC: {scores.mean():.4f} (+/- {scores.std():.4f})")
        self.metrics.update({'cv_scores': scores.tolist(), 'cv_mean': float(scores.mean()), 'cv_std': float(scores.std())})
        return scores

    def _log_results(self):
        """Выводит результаты."""
        logger.info(f"{'='*60}\nRESULTS: {self.model_name.upper()}\n{'='*60}")
        for split in ['train', 'test']:
            logger.info(f"{split.capitalize()}: " + ", ".join(f"{k}={v:.4f}" for k, v in self.metrics[split].items()))

        train_auc, test_auc = self.metrics['train']['roc_auc'], self.metrics['test']['roc_auc']
        overfit = train_auc - test_auc
        logger.info(f"Overfit: {overfit:.4f} (train={train_auc:.4f}, test={test_auc:.4f})")
        if overfit > 0.10:
            logger.warning("CRITICAL: Severe overfitting!")
        elif overfit > 0.05:
            logger.warning("Model is overfitting")

    @staticmethod
    def save_model(pipeline, model_name: str, metrics: dict):
        model_path = SAVED_MODELS_DIR / f"{model_name}_pipeline.joblib"
        metrics_path = RESULTS_DIR / f"{model_name}_metrics.joblib"
        joblib.dump(pipeline, model_path)
        joblib.dump(metrics, metrics_path)
        logger.info(f"Saved: {model_path}, {metrics_path}")
        return model_path, metrics_path

    @staticmethod
    def load_model(model_name: str) -> Tuple[Pipeline, Dict]:
        model_path = SAVED_MODELS_DIR / f"{model_name}_pipeline.joblib"
        metrics_path = RESULTS_DIR / f"{model_name}_metrics.joblib"
        return joblib.load(model_path), joblib.load(metrics_path) if metrics_path.exists() else {}


class SklearnTrainer(BaseTrainer):
    """Trainer для sklearn моделей."""
    MODELS = {
        'logistic_regression': LogisticRegression,
        'random_forest': RandomForestClassifier,
        'sgd_classifier': SGDClassifier,
    }

    def __init__(self, model_name: str, **kwargs):
        if model_name not in self.MODELS:
            raise ValueError(f"Unknown model: {model_name}. Available: {list(self.MODELS.keys())}")
        super().__init__(model_name=model_name, **kwargs)

    def _get_model(self):
        return self.MODELS[self.model_name](**self.model_params)


class LGBMTrainer(BaseTrainer):
    """Trainer для LightGBM."""
    def __init__(self, **kwargs):
        super().__init__(model_name='lightgbm', **kwargs)

    def _get_model(self):
        return LGBMClassifier(**self.model_params)


class XGBoostTrainer(BaseTrainer):
    """Trainer для XGBoost."""
    def __init__(self, **kwargs):
        super().__init__(model_name='xgboost', **kwargs)

    def _get_model(self):
        return XGBClassifier(**self.model_params)


class CatBoostTrainer(BaseTrainer):
    """Trainer для CatBoost."""
    def __init__(self, **kwargs):
        super().__init__(model_name='catboost', **kwargs)

    def _get_model(self):
        return CatBoostClassifier(**self.model_params)

    def _build_fit_kwargs(self, X_train, y_train, X_test, y_test) -> Dict:
        # Получаем preprocessor и трансформируем данные для определения категориальных колонок
        preprocessor = self._get_preprocessor()
        X_train_transformed = preprocessor.fit_transform(X_train, y_train)
        X_test_transformed = preprocessor.transform(X_test)

        # Определяем категориальные колонки после трансформации
        if isinstance(X_train_transformed, pd.DataFrame):
            cat_features = X_train_transformed.select_dtypes(include=['object', 'category']).columns.tolist()
        else:
            cat_features = []

        return {
            'model__cat_features': cat_features,
            'model__eval_set': (X_test_transformed, y_test)
        }


MODEL_TRAINERS = {
    'logistic_regression': SklearnTrainer,
    'random_forest': SklearnTrainer,
    'sgd_classifier': SklearnTrainer,
    'lightgbm': LGBMTrainer,
    'xgboost': XGBoostTrainer,
    'catboost': CatBoostTrainer,
}

AVAILABLE_MODELS = list(MODEL_TRAINERS.keys())

