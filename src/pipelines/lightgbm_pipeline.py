# src/pipelines/lightgbm_pipeline.py

from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

from src.pipelines.base_pipeline import BasePipelineBuilder
from src.config import NUMERICAL_FEATURES, CATEGORICAL_FEATURES, BIN_CATEGORICAL_FEATURES

class LightGBMPipelineBuilder(BasePipelineBuilder):
    """
    Cтроитель для LightGBMClassifier.
    Использует минимальный препроцессинг (только импутация).
    Логика FE берется из BasePipelineBuilder.
    """

    @staticmethod
    def _get_lgbm_preprocessor_pipe(self) -> Pipeline:
        """
        создает Pipeline, содержащий ColumnTransformer для минимальной предобработки LightGBM.
        выполняет только импутацию.
        """

        # 1. пайплайн для числовых (только импутация медианой)
        numerical_lgbm_pipe = Pipeline(
            steps=[('imputer', SimpleImputer(strategy='median'))]
        )

        # 2. пайплайн для категориальных (только импутация константой 'missing')
        categorical_lgbm_pipe = Pipeline(
            steps=[('imputer', SimpleImputer(strategy='constant', fill_value='missing'))]
        )

        cat_and_bin_features = CATEGORICAL_FEATURES + BIN_CATEGORICAL_FEATURES

        # ColumnTransformer
        lgbm_preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_lgbm_pipe, NUMERICAL_FEATURES),
                ('cat_and_bin', categorical_lgbm_pipe, cat_and_bin_features),
            ],
            remainder='drop',
            verbose_feature_names_out=False
        )

        # Инкапсулируем ColumnTransformer в Pipeline
        return Pipeline(steps=[('lgbm_imputer_ct', lgbm_preprocessor)])

    def _get_preprocessor(self, feature_engineering: bool = True) -> Pipeline:
        """
        собирает полный пайплайн препроцессинга: FE (если нужно) -> LGBM-препроцессор.
        """
        steps = []

        if feature_engineering:
            # 1. общие шаги feature engineering (из BasePipelineBuilder)
            steps.append(('feature_engineering', self._get_feature_engineering_pipeline()))

        # 2. специфичный для LightGBM препроцессор (только импутация)
        steps.append(('lgbm_preprocessor', self._get_lgbm_preprocessor_pipe()))

        return Pipeline(steps=steps)

    def _get_model(self) -> BaseEstimator:
        """
        создает инстанс LGBMClassifier, используя self.random_state из Base.
        """
        params = {'random_state': self.random_state}
        params.update(self.model_params)

        return LGBMClassifier(**params)