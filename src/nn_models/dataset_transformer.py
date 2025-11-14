import torch
from torch.utils.data import Dataset
import numpy as np

class CreditDataset(Dataset):
    """
    Пользовательский класс Dataset для PyTorch.
    Преобразует NumPy-массивы признаков (X) и меток (y) в тензоры.
    """
    def __init__(self, X: np.ndarray, y: np.ndarray):
        # Преобразование в тензоры float32 (для X) и float32 (для y)
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()

    def __len__(self):
        """Возвращает общее количество образцов."""
        return len(self.y)

    def __getitem__(self, idx):
        """Возвращает образец (X, y) по индексу."""
        # Для бинарной классификации метка должна иметь форму [1]
        return self.X[idx], self.y[idx].unsqueeze(-1)