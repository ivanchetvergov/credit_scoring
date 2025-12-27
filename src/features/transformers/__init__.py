# python
# File: src/features/transformers/__init__.py
import logging
from sklearn.base import BaseEstimator, TransformerMixin

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from .feature_creator import FeatureCreator
from .auxiliary_aggregator import AuxiliaryFeatureAggregator
from .dataframe_coercer import DataFrameCoercer
from .cv_target_encoder import CVTargetEncoder


class IdentityTransformer(BaseEstimator, TransformerMixin):
    """No-op transformer for pipeline compatibility."""
    def fit(self, X, y=None):
        return self

    @staticmethod
    def transform(X):
        return X


__all__ = [
    "FeatureCreator",
    "AuxiliaryFeatureAggregator",
    "DataFrameCoercer",
    "CVTargetEncoder",
    "IdentityTransformer",
]