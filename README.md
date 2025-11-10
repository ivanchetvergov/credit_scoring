# Кредитный дефолт:  ML-пайплайн

Проект для предсказания риска дефолта по кредиту на основе данных Home Credit. Этот пайплайн создан с фокусом на **воспроизводимость, масштабируемость** и достижение максимального ROC-AUC на сильно **несбалансированных данных**.

---

## 1. Описание и ключевые возможности

Этот проект реализует полный, MLOps-ориентированный ML-пайплайн для бинарной классификации:

* **Расширенный Feature Engineering:** агрегация данных из 5+ источников с созданием сотен значимых признаков.  
* **Архитектура "Trainer Pattern":** гибкая ООП-структура для инкапсуляции логики обучения и легкого добавления новых моделей.  
* **Обучение и тюнинг:** сравнение **6 моделей** (включая CatBoost, XGBoost, SGD Classifier), протюненных для несбалансированной классификации.  
* **Надежная оценка:** обязательная **Stratified K-Fold Cross-Validation** для подтверждения обобщающей способности (Generalization Score).

---

## 2. Архитектура: принцип "Trainer Pattern"

Ключевой элемент, обеспечивающий гибкость и надежность, — это **паттерн "Тренер"** (`BaseTrainer`).

| Аспект | Описание |
| :--- | :--- |
| **Инкапсуляция** | `BaseTrainer` управляет всем циклом: загрузкой конфигов, CV, логированием, сохранением артефактов. |
| **Специализация** | Дочерние классы (`CatBoostTrainer`, `XGBoostTrainer`) реализуют **только специфичную логику** (например, передачу `eval_set` или нативные механизмы обработки категориальных признаков). |
| **Динамический конфиг** | Все гиперпараметры для всех **6 моделей** **динамически подтягиваются** из централизованных файлов в папке `configs/`, обеспечивая **полную воспроизводимость**. |

---

## 3. Моделирование: битва ансамблей и линейной мощи

Пайплайн использует мульти-модельный подход, где каждая модель протюнена для борьбы с дисбалансом классов (используется `class_weight` или `scale_pos_weight`).

| Категория | Модели | Ключевой тюнинг и роль |
| :--- | :--- | :--- |
| **Бустинги (SOTA)** | `LightGBM`, `XGBoost`, `CatBoost` | **Максимальный ROC-AUC.** Используются нативные стратегии балансировки (`scale_pos_weight: 20`), `early_stopping` и тонкая настройка `learning_rate`. |
| **Линейные** | `Logistic Regression`, `SGD Classifier` | **Скорость и интерпретируемость.** Настроены на `L1/L2` регуляризацию. Идеальны для проверки Feature Selection и **самого быстрого инференса** в production. |
| **Ансамбль** | `Random Forest` | **Надежный бэйзлайн.** Тюнинг `max_depth` и `n_estimators` для сравнения с бустингами. |

---

## 4. Установка и использование

### 4.1. Требования

Для запуска требуется **Python 3.9+** (рекомендовано).

```bash
# 1. создать виртуальное окружение
python -m venv venv
source venv/bin/activate  # для Linux/Mac

# 2. установить зависимости
make install
# или
pip install -r requirements.txt
```

### 4.2. Подготовка данных

Поместите исходные CSV-файлы (application_train.csv, bureau.csv и т.д.) в директорию data/raw/.

### 4.3. Команды для запуска (The Main Script)

Основной цикл обучения и оценки управляется скриптом scripts/baseline_training.py.

| Сценарий              | Команда                                                | Назначение                                                       |
| :-------------------- | :----------------------------------------------------- | :--------------------------------------------------------------- |
| Полный цикл           | `make all`                                             | Full pipeline: Feature Engineering + обучение всех моделей с CV. |
| Полный бэйзлайн       | `python scripts/baseline_training.py`                  | Запускает все 6 моделей с 5-кратным CV.                          |
| Быстрый тест          | `python scripts/baseline_training.py --no-cv`          | Запускает все 6 моделей, но пропускает CV.                       |
| Фокусированный тюнинг | `python scripts/baseline_training.py catboost xgboost` | Обучение только указанных моделей.                               |

# 5. Структура проекта (расширенная)
```bash
.
├── Dockerfile
├── Makefile
├── README.md
├── app
│   ├── main.py
│   └── schemas.py
├── configs
│   ├── __init__.py
│   ├── catboost_config.py
│   ├── lightgbm_config.py
│   ├── processeed_features_config.py
│   ├── raw_features_config.py
│   ├── sklearn_config.py
│   └── xgboost_config.py
├── requirements.txt
├── results
│   ├── baseline_comparison.csv
│   ├── feature_order.json
│   ├── lightgbm_metrics.joblib
│   ├── logistic_regression_metrics.joblib
│   └── random_forest_metrics.joblib
├── scripts
│   ├── __init__.py
│   ├── feature_selection.py
│   ├── generate_features_config.py
│   └── train_model.py
├── src
│   ├── __init__.py
│   ├── config.py
│   ├── features
│   │   ├── __init__.py
│   │   ├── custom_transformers.py
│   │   ├── data_helpers.py
│   │   ├── feature_engineering.py
│   │   └── pipelines.py
│   ├── models
│   │   ├── __init__.py
│   │   ├── catboost_trainer.py
│   │   ├── lightgbm_trainer.py
│   │   ├── sklearn_trainer.py
│   │   ├── trainer_interface.py
│   │   └── xgboost_trainer.py
│   └── reporting
│       ├── __init__.py
│       └── compare_models.py
└── tests

10 directories, 37 files
```

# 6. Feature Engineering (сложные признаки)

- Проект создает более 400 признаков, используя агрегацию данных из нескольких источников.

- Агрегированные признаки: статистики (среднее, минимум, максимум) по Bureau, Bureau Balance и Previous Application.

- Расчетные признаки: отношения между ключевыми переменными (CREDIT_INCOME_RATIO, ANNUITY_INCOME_RATIO).

- Обработка аномалий: явное выделение аномальных значений (например, DAYS_EMPLOYED_ANOM).

# 7. Логирование, артефакты и качество
- Логирование

- Вся ключевая информация (загруженные параметры, CV-скоры, метрики Train/Test) логируется через стандартный модуль logging.

### Сохранение артефактов

- saved_models/ — полный пайплайн (препроцессор + модель) для production.

- results/ — метрики, CV-скоры, итоговая таблица сравнения моделей.
