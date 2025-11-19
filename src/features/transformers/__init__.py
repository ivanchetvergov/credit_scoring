# python
# File: src/features/transformers/__init__.py
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from .anomaly_handler import AnomalyHandler
from .feature_creator import FeatureCreator
from .auxiliary_aggregator import AuxiliaryFeatureAggregator
from .dataframe_coercer import DataFrameCoercer
from .cv_target_encoder import CVTargetEncoder
from sklearn.base import BaseEstimator, TransformerMixin

class IdentityTransformer(BaseEstimator, TransformerMixin):
    """Serializable no-op transformer."""
    def fit(self, X, y=None):
        return self

    @staticmethod
    def transform(X):
        return X

__all__ = [
    "AnomalyHandler",
    "FeatureCreator",
    "AuxiliaryFeatureAggregator",
    "DataFrameCoercer",
    "CVTargetEncoder",
    "IdentityTransformer",
]