# src/nn_models/__init__.py

from .torch_trainer import PyTorchTrainer

MODEL_TRAINERS_REGISTRY = {
    'simple_mlp': PyTorchTrainer,
}