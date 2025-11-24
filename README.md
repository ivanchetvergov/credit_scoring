# Кредитный дефолт — ML-пайплайн

Проект решает задачу бинарной классификации дефолта заемщика Home Credit. Репозиторий содержит полный цикл: от подготовки признаков до обучения нескольких моделей и сохранения артефактов для дальнейшего инференса.

## Архитектура и ключевые сущности

| Слой | Роль | Основные файлы |
| --- | --- | --- |
| Источник данных | Хранение исходных CSV Home Credit и подготовленных parquet | `data/raw/`, `data/processed/feature_store.parquet` |
| Feature engineering | Полное построение фичей, кастомные трансформеры и справочные функции | `src/features/feature_engineering.py`, `src/features/custom_transformers.py`, `src/features/data_helpers.py` |
| Пайплайны | Фабрика препроцессоров под каждую модель (категории, числовые признаки, бинарные флаги) | `src/pipelines/*.py`, функция `get_model_specific_pipeline` |
| Тренеры | Реализация паттерна `BaseTrainer`: CV, обучение, метрики, сохранение | `src/models/trainer_interface.py`, `src/models/*_trainer.py`, `src/models/__init__.py` |
| Конфигурации | Централизованные гиперпараметры и пути | `configs/*_config.py`, `src/config.py` |
| Скрипты и отчеты | CLI для обучения, генерации конфигов, сравнения результатов | `scripts/train_model.py`, `scripts/feature_selection.py`, `src/reporting/compare_models.py` |
| Сервинг | FastAPI для инференса обученных пайплайнов | `app/main.py`, `app/schemas.py` |

Основной сценарий: исходные CSV переносятся в `data/raw/`, затем `src/features/feature_engineering.py` строит feature store (`data/processed/feature_store.parquet`). Скрипт `scripts/train_model.py` по имени модели извлекает соответствующий тренер из `MODEL_TRAINERS_REGISTRY`, собирает `Pipeline(preprocessor, model)`, выполняет обучающий цикл (с опциональным Stratified K-Fold), сохраняет пайплайн и метрики (`saved_models/`, `results/`).

## Структура репозитория

```text
.
├── app/                 # FastAPI сервис для инференса
├── configs/             # Гиперпараметры для всех моделей
├── data/                # raw CSV и feature store
├── docker-compose.yaml  # docker-compose сценарий (обучение в контейнере)
├── Dockerfile           # Базовый образ Python 3.11 с запуском train_model.py
├── Makefile             # Автоматизация (features, train, lint, docker)
├── scripts/             # CLI-инструменты (train_model.py, generate_features_config.py, ...)
├── src/                 # Основная библиотека: features, pipelines, models, reporting
├── saved_models/        # Сохраненные пайплайны (joblib)
├── results/             # Метрики CV/Test и сравнение моделей
└── app/                 # FastAPI приложение (инференс)
```

## Требования и подготовка данных

1. Python 3.9+ (Docker-образ собирается на Python 3.11).
2. Исходные CSV из датасета Home Credit должны быть скопированы в `data/raw/` (минимум `application_train.csv`; для расширенного feature engineering — остальные CSV из каталога `data/raw/`).
3. Структура каталогов создается автоматически командой `make setup` (создаст `data`, `saved_models`, `results`, `logs`).

## Локальный запуск (без Docker)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Проверить наличие исходных данных
make data

# Построить feature store (data/processed/feature_store.parquet)
make features

# Обучить все зарегистрированные модели с CV
make train

# Примеры параметров
make train MODEL=catboost          # только CatBoost-тренер
make train NO_CV=1                 # ускоренное обучение без кросс-валидации
python scripts/train_model.py catboost simple_mlp --no-cv --no-compare
```

Результаты обучения записываются в `saved_models/{model}_pipeline.joblib` и `results/{model}_metrics.joblib`. Итоговая сводка ROC-AUC и других метрик сохраняется в `results/baseline_comparison.csv`.

## Архитектурные детали тренеров

- `BaseTrainer` (см. `src/models/trainer_interface.py`) отвечает за загрузку параметров из `MODEL_PARAMS`, подготовку `Pipeline`, Stratified K-Fold, расчет метрик, извлечение feature importance и сохранение артефактов.
- Конкретные тренеры (`CatBoostTrainer`, `LGBMTrainer`, `SklearnTrainer`, `PyTorchTrainer`) реализуют `_get_model()` и доп. fit-аргументы (eval_set, class_weight и т.д.).
- `get_model_specific_pipeline()` (в `src/pipelines/__init__.py`) возвращает подходящий `ColumnTransformer` с учетом типа модели (например, CatBoost не требует полного OHE).
- Hyperparameters определяются в `configs/*_config.py` и агрегируются в `MODEL_PARAMS` (`src/config.py`).

## Команды Makefile

| Команда | Описание |
| --- | --- |
| `make setup` | Создает структуру каталогов и устанавливает зависимости (совместно с `make install`). |
| `make features` | Запускает `python -m src.features.feature_engineering`, создает/обновляет feature store. |
| `make train [MODEL=<name>] [NO_CV=1]` | Обучает модели через `scripts.train_model` (по умолчанию все из `MODEL_TRAINERS_REGISTRY`). |
| `make all` | Полный цикл: очистка, feature engineering, обучение. |
| `make clean` | Удаляет кэш, обработанные данные, сохраненные модели/результаты. |
| `make docker-build`, `make docker-run` | Сборка и запуск контейнера вручную. |
| `make run-api` | Запуск FastAPI сервера для инференса (требует предварительно обученной модели). |

## Запуск в Docker

### Вариант 1. Docker CLI

```bash
# Сборка образа
docker build -t credit-default-ml .

# Запуск обучения (по умолчанию train_model.py --no-cv)
docker run --rm \
  -v "$(pwd)/data":/app/data \
  -v "$(pwd)/results":/app/results \
  -v "$(pwd)/saved_models":/app/saved_models \
  credit-default-ml
```

Контейнер использует `CMD ["python", "-u", "scripts/train_model.py", "--no-cv"]`. При необходимости можно переопределить команду, например: `docker run ... credit-default-ml python scripts/train_model.py catboost`.

### Вариант 2. docker-compose

`docker-compose.yaml` описывает сервис `ml_trainer`:

```bash
docker compose up --build
```

Volumes монтируют каталоги `data`, `results`, `configs`, поэтому модельные артефакты и feature store остаются на хосте. Для повторного обучения достаточно обновить конфиги, перезапустить `docker compose up --build`.

## FastAPI инференс

После обучения нужной модели:

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Эндпоинт `/predict` принимает структуру из `app/schemas.py`. Перед запуском убедитесь, что `saved_models/<model>_pipeline.joblib` доступен сервису (или задайте переменные окружения, если модифицируете app/main.py).

## Логи и артефакты

- Feature engineering пишет промежуточные логи через стандартный `logging` и формирует `data/processed/feature_store.parquet`.
- Обучение фиксирует параметры, ROC-AUC и конфьюжн-матрицы; результаты доступны в `results/*.joblib` и `results/baseline_comparison.csv`.
- `BaseTrainer.save_model` гарантирует согласованность между пайплайном и метриками; загрузить обученную модель можно через `BaseTrainer.load_model(model_name)`.

## Дополнительные материалы

- `EDA/` — ноутбуки с исследованием категориальных и числовых признаков.
- `scripts/feature_selection.py` — вспомогательный скрипт для отбора признаков.
- `src/reporting/compare_models.py` — генерация сравнительных таблиц и визуализаций.

Все изменения архитектуры (новые модели, иные фичи) рекомендуется вносить через обновление соответствующих тренеров и конфигов, чтобы сохранить совместимость с единым CLI и Docker-пайплайном.
