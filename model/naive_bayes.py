from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer


def _to_dense_if_sparse(features):
    """Convert sparse matrices to dense arrays for GaussianNB compatibility."""
    return features.toarray() if hasattr(features, "toarray") else features


def build_naive_bayes() -> Pipeline:
    """Build GaussianNB with a sparse->dense adapter for one-hot encoded inputs."""
    return Pipeline(
        steps=[
            ("to_dense", FunctionTransformer(_to_dense_if_sparse, accept_sparse=True)),
            ("gaussian_nb", GaussianNB()),
        ]
    )