.PHONY: help install clean lint test data features train all setup info

# ==============================================================================
# КОНФИГУРАЦИЯ И УТИЛИТЫ
# ==============================================================================
# Цвета для вывода
BLUE := \033[0;34m
GREEN := \033[0;32m
RED := \033[0;31m
NC := \033[0m # No Color

PYTHON_VENV := .venv/bin/python
SCRIPTS_DIR := scripts
MODULE_DIR := src

# ==============================================================================
# ГЛАВНЫЕ КОМАНДЫ (PIPELINE)
# ==============================================================================

help: ## Показать это сообщение помощи
	@echo "$(BLUE)Доступные команды:$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-18s$(NC) %s\n", $$1, $$2}'
	@echo "$(BLUE)Параметры для 'train':$(NC)"
	@echo "  $(GREEN)make train MODEL=lightgbm$(NC)  Обучить конкретную модель"
	@echo "  $(GREEN)make train NO_CV=1$(NC)	  Обучить без кросс-валидации"

features: data ## Создать/обновить feature store. Зависит от 'data'
	@echo "$(BLUE)Запуск feature engineering...$(NC)"
	$(PYTHON_VENV) -m $(MODULE_DIR).features.feature_engineering
	@echo "$(GREEN)  Feature store создан$(NC)"

train: ## Обучить модели (по умолчанию все). Параметры: MODEL, NO_CV
	@echo "$(BLUE)Запуск обучения baseline моделей ($(MODEL))...$(NC)"
	$(PYTHON_VENV) -m $(SCRIPTS_DIR).baseline_training $(MODEL) $(if $(NO_CV),--no-cv,)
	@echo "$(GREEN)  Обучение завершено$(NC)"

all: clean features train ## Полный цикл: очистка -> feature engineering -> обучение
	@echo "$(GREEN)  Полный pipeline завершен$(NC)"

# ==============================================================================
# ИНФРАСТРУКТУРНЫЕ КОМАНДЫ
# ==============================================================================

install: ## Установить зависимости
	@echo "$(BLUE)Установка зависимостей...$(NC)"
	pip install -r requirements.txt
	@echo "$(GREEN)  Зависимости установлены$(NC)"

setup: install ## Первоначальная настройка директорий проекта
	@echo "$(BLUE)Настройка проекта...$(NC)"
	mkdir -p data/raw data/processed saved_models results logs
	touch data/raw/.gitkeep data/processed/.gitkeep
	@echo "$(GREEN)  Проект настроен$(NC)"

# ==============================================================================
# КОМАНДЫ ОЧИСТКИ
# ==============================================================================

clean: clean-cache clean-data clean-models ## Полная очистка: кэш, данные, модели
	@echo "$(GREEN)  Полная очистка завершена$(NC)"

clean-cache: ## Очистить временные файлы и кэш (pyc, __pycache__, pytest)
	@echo "$(BLUE)Очистка кэша...$(NC)"
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .mypy_cache 2>/dev/null || true
	@echo "$(GREEN)  Кэш удален$(NC)"

clean-data: ## Очистить обработанные данные
	@echo "$(BLUE)Очистка обработанных данных...$(NC)"
	rm -rf data/processed/* 2>/dev/null || true
	touch data/processed/.gitkeep
	@echo "$(GREEN)  Обработанные данные удалены$(NC)"

clean-models: ## Очистить сохраненные модели и результаты
	@echo "$(BLUE)Очистка моделей и результатов...$(NC)"
	rm -rf saved_models/* results/* 2>/dev/null || true
	mkdir -p saved_models results # Создаем, чтобы не было ошибок при train
	@echo "$(GREEN)  Модели и результаты удалены$(NC)"

# ==============================================================================
# КОМАНДЫ КАЧЕСТВА КОДА
# ==============================================================================

lint: ## Проверить код с помощью flake8
	@echo "$(BLUE)Проверка кода (lint)...$(NC)"
	flake8 $(MODULE_DIR)/ $(SCRIPTS_DIR)/ --max-line-length=100 --ignore=E501,W503
	@echo "$(GREEN)  Проверка завершена$(NC)"

test: ## Запустить тесты
	@echo "$(BLUE)Запуск тестов...$(NC)"
	$(PYTHON_VENV) -m pytest tests/ -v
	@echo "$(GREEN)  Тесты завершены$(NC)"

# ==============================================================================
# УДОБСТВО И ИНФО
# ==============================================================================

data: ## Проверить наличие исходных данных
	@if [ ! -f data/raw/application_train.csv ]; then \
	echo "$(RED)  Исходные данные (application_train.csv) не найдены в data/raw.$(NC)"; \
	exit 1; \
	fi
	@echo "$(GREEN)  Исходные данные найдены$(NC)"

info: ## Показать информацию о проекте
	@echo "$(BLUE)Информация о проекте:$(NC)"
	@echo "  Python версия: $$(python --version)"
	@echo "  Директории:"
	@echo "	- data/raw: $$(ls data/raw 2>/dev/null | wc -l) файлов"
	@echo "	- data/processed: $$(ls data/processed 2>/dev/null | wc -l) файлов"
	@echo "	- saved_models: $$(ls saved_models 2>/dev/null | wc -l) файлов"
	@echo "	- results: $$(ls results 2>/dev/null | wc -l) файлов"