from abc import ABC
from typing import Dict, Any, Tuple, Optional
from torch.utils.data import DataLoader

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging

from src.models.trainer_interface import BaseTrainer
from src.nn_models.simple_mlp import SimpleMLP
from src.nn_models.dataset_transformer import CreditDataset
from src.pipelines import get_model_specific_pipeline

logger = logging.getLogger(__name__)

import os

# КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: Отключение неявного параллелизма
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

class PyTorchTrainer(BaseTrainer, ABC):

    def __init__(self, model_name: str, model_params: Optional[Dict[str, Any]] = None, **kwargs):
        super().__init__(model_name, model_params, **kwargs)

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        # Нам нужно маппирование, чтобы выбирать класс модели
        self.model_map = {'simple_mlp': SimpleMLP}
        self.model_class = self.model_map.get(model_name)
        self.model = None

    def _initialize_pytorch_components(self, input_size: int):
        # 1. Модель
        if self.model_class:
            self.model = self.model_class(self.model_params, input_size).to(self.device)
            self.model.print_model_summary()
        else:
            raise ValueError(f"Unknown PyTorch model: {self.model_name}")

        # 2. Функция Потерь (Loss)
        pos_weight = torch.tensor([self.model_params.get('pos_weight', 1.0)], device=self.device)
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        # 3. Оптимизатор
        optimizer_name = self.model_params.get('optimizer', 'Adam')
        lr = self.model_params.get('learning_rate', 0.001)

        if optimizer_name == 'Adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        else:
            raise NotImplementedError(f"Optimizer {optimizer_name} not supported yet.")

    def train(self, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, **kwargs) -> \
    Tuple[Any, Dict[str, float]]:
        logger.info(f"Training {self.model_name} model on device {self.device}...")

        preprocessor = get_model_specific_pipeline(
            model_name='lightgbm',  # Используем препроцессор для LightGBM
            include_feature_engineering=True
        )

        logger.info("Applying LightGBM preprocessor (scaling, encoding, imputation)...")

        # 2. Трансформируем данные
        X_train_processed = preprocessor.fit_transform(X_train, y_train)
        X_test_processed = preprocessor.transform(X_test)

        # y_train и y_test должны быть преобразованы в float32
        y_train_np = y_train.values.astype(np.float32)
        y_test_np = y_test.values.astype(np.float32)

        input_size = X_train_processed.shape[1]  # Используем размерность обработанных данных
        self._initialize_pytorch_components(input_size)

        # 3. Создание DataLoader'ов
        train_dataset = CreditDataset(X_train_processed, y_train_np)
        test_dataset = CreditDataset(X_test_processed, y_test_np)

        batch_size = self.model_params['batch_size']
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0
        )

        best_auc = 0.0
        patience_counter = 0

        for epoch in range(1, self.model_params['n_epochs'] + 1):
            # --- Тренировочный Цикл (сокращен для обзора) ---
            self.model.train()
            running_loss = 0.0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(X_batch)

                y_batch = y_batch.squeeze(-1)

                loss = self.criterion(outputs, y_batch)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item() * X_batch.size(0)

            # --- Валидационный Цикл ---
            val_loss, val_metrics = self._validate(test_loader)

            # --- Логика Ранней Остановки ---
            current_auc = val_metrics['Test ROC-AUC']
            if epoch % 100 == 0:
                logger.info(
                    f"Epoch {epoch}: Train Loss: {running_loss / len(train_dataset):.4f}, Val AUC: {current_auc:.4f}")

            if current_auc > best_auc:
                best_auc = current_auc
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.model_params.get('early_stopping_patience', 10):
                    logger.info(f"Early stopping triggered at epoch {epoch}")
                    break

        final_metrics = self._evaluate_final(test_loader)
        return self.model, final_metrics

    def _validate(self, data_loader: DataLoader) -> Tuple[float, Dict[str, float]]:
        self.model.eval()
        running_loss = 0.0
        all_labels = []
        all_predictions = []

        with torch.no_grad():
            for X_batch, y_batch in data_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                outputs = self.model(X_batch)

                # Сжимаем таргет перед расчетом потерь
                y_batch_squeezed = y_batch.squeeze(-1)

                # Используем сжатый таргет для расчета потерь
                running_loss += self.criterion(outputs, y_batch_squeezed).item() * X_batch.size(0)

                all_labels.append(y_batch.cpu().numpy())
                all_predictions.append(outputs.sigmoid().cpu().numpy())

        val_loss = running_loss / len(data_loader.dataset)

        y_true = np.concatenate(all_labels).flatten()
        y_pred_proba = np.concatenate(all_predictions).flatten()

        # Создаем бинарные метки y_pred
        y_pred = (y_pred_proba > 0.5).astype(int)

        # 1. Рассчитываем метрики, используя ключи BaseTrainer
        base_metrics = self._calculate_metrics(y_true, y_pred, y_pred_proba)

        metrics = {
            'Test ROC-AUC': base_metrics['roc_auc'],
            'Test Accuracy': base_metrics['accuracy'],
        }

        return val_loss, metrics

    def _evaluate_final(self, data_loader: DataLoader) -> Dict[str, float]:
        _, final_metrics = self._validate(data_loader)
        final_metrics['Train ROC-AUC'] = np.nan  # Требует отдельного прохода
        final_metrics['CV Mean'] = np.nan
        final_metrics['CV Std'] = np.nan
        return final_metrics

    def _get_model(self):
        """
        Заглушка для удовлетворения требования абстрактного класса BaseTrainer.
        Модель PyTorch инициализируется в методе _initialize_pytorch_components.
        """
        # Возвращаем None или просто поднимаем исключение, чтобы не использовать его
        # в PyTorchTrainer. Логика инициализации модели находится в self._initialize_pytorch_components.
        return None
