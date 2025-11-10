# src/models/torch_models/base_model.py

import torch
import torch.nn as nn
from typing import Dict, Any

class BaseModel(nn.Module):
    """"
    Абстрактный базовый класс для всех нейронных сетей на PyTorch.
    Наследует torch.nn.Module и задает стандарты инициализации и подсчета параметров.
    """

    def __init__(self, model_params: Dict[str, Any], input_size: int):
        """
                :param model_params: Словарь гиперпараметров модели (например, слои, активации).
                :param input_size: Размерность входного вектора (количество признаков).
                """
        super().__init__()
        self.model_params = model_params
        self.input_size = input_size

        self.model = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Прямой проход (forward pass) через сеть.
        Этот метод должен быть переопределен в дочернем классе.

        :param x: Входной тензор признаков.
        :return: Выходной тензор (например, логиты или вероятности).
        """
        raise NotImplementedError("Метод 'forward' должен быть переопределен в дочернем классе.")

    def _get_model(self) -> nn.Module:
        """
        Внутренний метод для создания архитектуры сети.
        Должен быть переопределен в дочернем классе.
        """
        raise NotImplementedError("Метод '_get_model' должен быть переопределен в дочернем классе.")

    def print_model_summary(self) -> None:
        """
        Выводит краткую сводку по модели.
        """
        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print("=" * 50)
        print(f"PyTorch Model Summary: {self.__class__.__name__}")
        print(f"Input Size: {self.input_size}")
        print(f"Trainable parameters: {n_params:,}")
        print("=" * 50)

    # добавляем стандартный метод для загрузки весов
    def load_weights(self, path: str) -> None:
        """Загружает предобученные веса."""
        self.load_state_dict(torch.load(path))
        print(f"Weights loaded from {path}")