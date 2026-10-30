"""Count-vector anomaly detectors. Every model exposes fit(X, y) and score(X),
where a higher score means more anomalous."""
import numpy as np
from scipy import sparse
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from src.config import RANDOM_STATE


def _log_counts(X) -> sparse.csr_matrix:
    X = sparse.csr_matrix(X, dtype=np.float64, copy=True)
    np.log1p(X.data, out=X.data)
    return X


class PCADetector:
    """Xu et al. (SOSP'09): project TF-IDF weighted, mean-centred counts onto the principal
    subspace of normal behaviour; the anomaly score is the squared residual (SPE / Q-statistic)."""
    name = "PCA"
    supervised = False

    def __init__(self, variance: float = 0.95):
        self.variance = variance

    def _weight(self, X) -> np.ndarray:
        return np.asarray(X.multiply(self.idf_).todense()) - self.mean_

    def fit(self, X, y=None) -> "PCADetector":
        n = X.shape[0]
        doc_freq = np.asarray((X > 0).sum(axis=0)).ravel()
        self.idf_ = np.log(n / np.maximum(doc_freq, 1))
        self.mean_ = np.asarray(X.multiply(self.idf_).mean(axis=0)).ravel()
        self.pca_ = PCA(n_components=self.variance, svd_solver="full").fit(self._weight(X))
        return self

    def score(self, X) -> np.ndarray:
        Z = self._weight(X)
        residual = Z - self.pca_.inverse_transform(self.pca_.transform(Z))
        return np.einsum("ij,ij->i", residual, residual)


class IsolationForestDetector:
    name = "Isolation Forest"
    supervised = False

    def __init__(self, n_estimators: int = 200):
        self.model = IsolationForest(n_estimators=n_estimators, random_state=RANDOM_STATE, n_jobs=-1)

    def fit(self, X, y=None) -> "IsolationForestDetector":
        self.model.fit(_log_counts(X))
        return self

    def score(self, X) -> np.ndarray:
        return -self.model.score_samples(_log_counts(X))


class LogisticRegressionDetector:
    name = "Logistic Regression (supervised)"
    supervised = True

    def __init__(self):
        self.model = LogisticRegression(class_weight="balanced", max_iter=2000)

    def fit(self, X, y) -> "LogisticRegressionDetector":
        self.model.fit(_log_counts(X), y)
        return self

    def score(self, X) -> np.ndarray:
        return self.model.predict_proba(_log_counts(X))[:, 1]


class DecisionTreeDetector:
    name = "Decision Tree (supervised)"
    supervised = True

    def __init__(self):
        self.model = DecisionTreeClassifier(class_weight="balanced", min_samples_leaf=5,
                                            random_state=RANDOM_STATE)

    def fit(self, X, y) -> "DecisionTreeDetector":
        self.model.fit(X, y)
        return self

    def score(self, X) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]


def default_models() -> list:
    return [PCADetector(), IsolationForestDetector(), LogisticRegressionDetector(), DecisionTreeDetector()]
