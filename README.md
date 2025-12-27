# Кредитный дефолт — ML-пайплайн

Проект решает задачу бинарной классификации дефолта заемщика Home Credit.

## Архитектура

| Слой | Роль | Файлы |
| --- | --- | --- |
| Данные | Исходные CSV и feature store | `data/raw/`, `data/processed/feature_store.parquet` |
| Feature engineering | Трансформеры и агрегации | `src/features/transformers/`, `src/features/feature_engineering.py` |
| Пайплайны | Препроцессоры для каждой модели | `src/pipelines/preprocessing.py` |
| Тренеры | `BaseTrainer` и специализированные классы | `src/models/trainers.py` |
| Конфигурации | Гиперпараметры и пути | `configs/`, `src/config.py` |
| Скрипты | CLI для обучения и анализа | `scripts/train_model.py` |
| Сервинг | FastAPI для инференса | `app/` |

## Быстрый старт

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Построить feature store
make features

# Обучить все модели
make train

# Или конкретную модель без CV
python -m scripts.train_model catboost --no-cv
```

## Доступные модели

- `logistic_regression`, `random_forest`, `sgd_classifier` (sklearn)
- `lightgbm`, `xgboost`, `catboost` (gradient boosting)

## Структура

```
├── app/                 # FastAPI сервис
├── configs/             # Гиперпараметры моделей
├── data/                # raw CSV и feature store
├── scripts/             # CLI (train_model.py, feature_selection.py)
├── src/
│   ├── config.py        # Пути и MODEL_PARAMS
│   ├── features/        # Трансформеры (FeatureCreator, AuxiliaryAggregator)
│   ├── models/          # trainers.py (BaseTrainer, SklearnTrainer, etc.)
│   ├── pipelines/       # preprocessing.py
│   └── reporting/       # compare_models.py
├── saved_models/        # joblib пайплайны
└── results/             # Метрики и сравнения
```

## Архитектура тренеров

- `BaseTrainer` — базовый класс: загрузка параметров, построение Pipeline, CV, метрики, сохранение
- `SklearnTrainer` — для sklearn моделей (LogisticRegression, RandomForest, SGD)
- `LGBMTrainer`, `XGBoostTrainer`, `CatBoostTrainer` — для gradient boosting

Pipeline: `[preprocessor] → [model]`
- Для sklearn/lgbm/xgboost: `DataFrameCoercer → FeatureCreator → ColumnTransformer`
- Для CatBoost: `CatBoostPreprocessor` (FE + подготовка категорий)

## Docker

```bash
docker compose up --build
```

## FastAPI

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## Команды Makefile

| Команда | Описание |
| --- | --- |
| `make features` | Создает feature store |
| `make train` | Обучает все модели с CV |
| `make train MODEL=catboost NO_CV=1` | Обучает CatBoost без CV |
| `make clean` | Очистка кэша и результатов |
