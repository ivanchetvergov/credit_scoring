<!-- Repo-specific Copilot instructions for code agents -->
# Copilot / AI agent quick guide — credit_scoring

Purpose: Help AI coding agents become productive quickly in this repo (ML pipeline for credit default prediction).

## 1) Big picture
- End-to-end ML pipeline: raw CSVs → feature store → model training → saved pipelines and metrics.
- Core pattern: "Trainer Pattern". All trainers in `src/models/trainers.py` (BaseTrainer, SklearnTrainer, LGBMTrainer, XGBoostTrainer, CatBoostTrainer).
- Feature engineering in `src/features/`, pipeline factory in `src/pipelines/preprocessing.py`.

## 2) Key files
- `data/raw/` — original CSVs (application_train.csv, bureau.csv, ...)
- `data/processed/feature_store.parquet` — feature store (`FEATURE_STORE_PATH` in `src/config.py`)
- `src/features/transformers/` — FeatureCreator, AuxiliaryFeatureAggregator, CVTargetEncoder
- `src/pipelines/preprocessing.py` — `get_preprocessing_pipeline(model_name)`
- `src/models/trainers.py` — all trainer classes and MODEL_TRAINERS registry
- `configs/raw_features_config.py` — feature lists (NUMERICAL, CATEGORICAL, BINARY)
- `src/config.py` — MODEL_PARAMS, paths, constants
- `scripts/train_model.py` — CLI for training

## 3) Data flow
1. Feature engineering: `python -m src.features.feature_engineering` → feature_store.parquet
2. Training: `python -m scripts.train_model` → loads feature store, trains models, saves pipelines

## 4) Patterns
- Trainer: implement `_get_model()` in subclass, optionally override `_build_fit_kwargs()`
- Pipeline: `[preprocessor] → [model]`
  - sklearn/lgbm/xgboost: DataFrameCoercer → FeatureCreator → ColumnTransformer
  - catboost: CatBoostPreprocessor (handles categorical columns)
- Config-first: hyperparams in `src/config.py` under `MODEL_PARAMS`
- Persistence: `saved_models/{model}_pipeline.joblib`, `results/{model}_metrics.joblib`

## 5) How to run
```bash
make features          # Create feature store
make train             # Train all models with CV
python -m scripts.train_model catboost --no-cv  # Single model, no CV
```

## 6) Debugging
- Missing features error: check `FEATURE_STORE_PATH` exists
- Model not training: check `MODEL_TRAINERS` dict in `src/models/trainers.py`
- Adding new model: add trainer class, add to MODEL_TRAINERS, add params to MODEL_PARAMS

## 7) Tests
- `tests/` folder exists but no tests yet. Run `pytest` when tests are added.
