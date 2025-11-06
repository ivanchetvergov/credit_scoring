import joblib
import logging
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
)
from sklearn.pipeline import Pipeline

from src.config import (
    MODELS_DIR,
    RESULTS_DIR,
    SEED,
    CV_FOLDS,
    MODEL_PARAMS
)
from src.features.features import get_preprocessing_pipeline

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BaselineTrainer:
    """
    Класс для обучения и оценки baseline моделей.
    """

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.pipeline = None
        self.metrics = {}

        # создаем директории
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    def _get_model(self):
        """
        Возращает модель по имени.
        :return:
        """
        models = {
            'logistic_regression': LogisticRegression(**MODEL_PARAMS['logistic_regression']),
            'random_forest': RandomForestClassifier(**MODEL_PARAMS['random_forest'])
        }

        if self.model_name not in models:
            raise ValueError(f"Unknown model: {self.model_name}")

        return models[self.model_name]

    @staticmethod
    def _calculate_metrics(y_true, y_pred, y_pred_proba):
        """Вычисляет метрики качества модели"""
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            "recall": recall_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred),
            'roc_auc': roc_auc_score(y_true, y_pred_proba)
        }

    def train(self, X_train, y_train, X_test, y_test):
        """
        Обучает модель и оценивает на тестовой выборке
        """
        logger.info('=' * 80)
        logger.info(f"Training {self.model_name} model...")
        logger.info('=' * 80)

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

        # предсазание на train
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
        self.metrics = {
            'train': train_metrics,
            'test': test_metrics
        }

        # логируем результаты
        self._log_results()

        # сохраняем confusion matrix
        cm = confusion_matrix(y_test, y_test_pred)
        logger.info(f"Confusion matrix: {cm}")

        return self.pipeline, self.metrics

    def cross_validate(self, X, y):
        """Выполняет кросс-валидацию"""
        logger.info(f"\nPerforming {CV_FOLDS}-fold cross-validation...")

        cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)

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
            n_jobs=-1
        )

        logger.info(f"Cross-validation ROC-AUC scores: {cv_scores}")
        logger.info(f"Mean CV ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

        self.metrics['cv_scores'] = cv_scores
        self.metrics['cv_mean'] = cv_scores.mean()
        self.metrics['cv_std'] = cv_scores.std()

        return cv_scores

    def _log_results(self):
        """Логирует результаты обучения."""
        logger.info("=" * 80)
        logger.info(f"RESULTS FOR {self.model_name.upper()}")
        logger.info("=" * 80)

        logger.info("\nTrain Metrics:")
        for metric, value in self.metrics['train'].items():
            logger.info(f"  {metric}: {value:.4f}")

        logger.info("\nTest Metrics:")
        for metric, value in self.metrics['test'].items():
            logger.info(f"  {metric}: {value:.4f}")

        # проверка на overfitting
        train_auc = self.metrics['train']['roc_auc']
        test_auc = self.metrics['test']['roc_auc']
        overfit = train_auc - test_auc

        logger.info(f"\nOverfitting check:")
        logger.info(f"  Train ROC-AUC: {train_auc:.4f}")
        logger.info(f"  Test ROC-AUC: {test_auc:.4f}")
        logger.info(f"  Difference: {overfit:.4f}")

        if overfit > 0.06:
            logger.warning("  Model might be overfitting!")
        else:
            logger.info("  Model generalization looks good")

    @staticmethod
    def save_model(pipeline, model_name: str, metrics: dict):
        """Сохраянет обученую модель"""
        # убедимся, что директории существуют
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)

        # сохраняем модель
        model_path = MODELS_DIR / f"{model_name}_pipeline.joblib"
        joblib.dump(pipeline, model_path)
        logger.info(f"Saved model pipeline to {model_path}")

        # сохраняем метрики
        metrics_path = RESULTS_DIR / f"{model_name}_metrics.joblib"
        joblib.dump(metrics, metrics_path)
        logger.info(f"Saved metrics to {metrics_path}")