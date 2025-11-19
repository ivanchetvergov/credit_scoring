# src/pipelines/lightgbm_pipeline.py

from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from src.config import NUMERICAL_FEATURES, CATEGORICAL_FEATURES, BIN_CATEGORICAL_FEATURES
from src.pipelines.base_pipeline import BasePipelineBuilder
from src.pipelines.base_preprocessor import get_base_feature_engineering_steps, get_binary_encoder

class LightGBMPipelineBuilder(BasePipelineBuilder):
    """
    Cтроитель для LightGBMClassifier.
    Использует минимальный препроцессинг (только импутация).
    Логика FE берется из BasePipelineBuilder.
    """
    def _get_preprocessor(self, feature_engineering: bool = True) -> Pipeline:
        """
        собирает полный пайплайн препроцессинга: FE (если нужно) -> LGBM-препроцессор.
        """
        return get_lightgbm_preprocessing_pipeline(
            include_feature_engineering=feature_engineering
        )


    def _get_model(self) -> BaseEstimator:
        """
        создает инстанс LGBMClassifier, используя self.random_state из Base.
        """
        params = {'random_state': self.random_state}
        params.update(self.model_params)

        return LGBMClassifier(**params)


def get_lightgbm_preprocessing_pipeline(include_feature_engineering: bool = True) -> Pipeline:
    """
    Создает минимальный пайплайн препроцессинга для LightGBM.
    Числовые: только импутация. Категории: импутация + OrdinalEncoder.
    """
    # logger.info(f"Building LightGBM pipeline (FE={'enabled' if include_feature_engineering else 'disabled'})")

    steps = []
    if include_feature_engineering:
        steps.extend(get_base_feature_engineering_steps())

    # Числовые: только медианная импутация (без масштабирования)
    num_pipe = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median'))
    ])

    # Категориальные (многозначные): импутация + OrdinalEncoder (LGBM лучше работает с числами)
    cat_pipe = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
    ])

    column_transformer = ColumnTransformer(
        transformers=[
            ('num', num_pipe, NUMERICAL_FEATURES),
            ('cat', cat_pipe, CATEGORICAL_FEATURES),
            ('bin_enc', get_binary_encoder(), BIN_CATEGORICAL_FEATURES) # Используем кодировщик
        ],
        remainder='drop',
        verbose_feature_names_out=False
    )

    steps.append(('preprocessor', column_transformer))
    return Pipeline(steps=steps)
