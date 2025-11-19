from src.config import SEED, SCALE_POS_WEIGHT

MODEL_PARAMS = {
    'simple_mlp': {
        # Архитектура
        'hidden_units': [256, 128, 64],  # Размер скрытых слоев
        'dropout': 0.1,  # Коэффициент Dropout

        # Тренировка
        'n_epochs': 50,  # Количество эпох
        'batch_size': 256,  # Размер батча (чем больше, тем быстрее, но требует больше памяти)
        'learning_rate': 0.001,  # Скорость обучения для оптимизатора Adam
        'optimizer': 'Adam',  # Оптимизатор
        'loss_function': 'BCEWithLogitsLoss',  # Функция потерь для бинарной классификации

        # Обработка дисбаланса (весовой коэффициент для положительного класса)
        'pos_weight': SCALE_POS_WEIGHT,

        'random_state': SEED,
        'early_stopping_patience': 10,  # Пациентность для ранней остановки
    }
}