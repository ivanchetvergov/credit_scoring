import torch
import torch.nn as nn
from typing import Dict, Any

from src.nn_models.base_model import BaseModel

class SimpleMLP(BaseModel):
    """
    Простая многослойная нейронная сеть (MLP) для бинарной классификации.
    """

    def __init__(self, model_params: Dict[str, Any], input_size: int):
        super().__init__(model_params, input_size)
        self.hidden_units = model_params.get('hidden_units', [128, 64])
        self.dropout_rate = model_params.get('dropout', 0.1)

        # создаем модель и сохраняем ее в self.model
        self.model = self._get_model()

    def _get_model(self) -> nn.Sequential:
        """
        Создает архитектуру сети динамически.
        """
        layers = []
        current_in_size = self.input_size

        for units in self.hidden_units:
            layers.append(nn.Linear(current_in_size, units))
            layers.append(nn.BatchNorm1d(units))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(self.dropout_rate))
            current_in_size = units

        # выходной слой: 1 нейрон (логиты)
        layers.append(nn.Linear(current_in_size, 1))

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Прямой проход. Возвращает логиты.
        """
        # гарантируем, что входные данные имеют тип float
        x = x.float()

        logits = self.model(x)

        # изменяем форму: [batch_size, 1] -> [batch_size]
        return logits.squeeze(-1)