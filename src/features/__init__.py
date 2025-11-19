from sklearn.base import BaseEstimator, TransformerMixin


class IdentityTransformer(BaseEstimator, TransformerMixin):
    """Сериализуемый трансформер-заглушка. Возвращает входные данные."""
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X
