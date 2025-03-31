import numpy as np
from sklearn.metrics import silhouette_score, mean_squared_error

class CustomKMeans:
    def __init__(self, n_clusters=8, max_iter=300, tol=1e-4, distance_matrix=None, random_state=42):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.distance_matrix = distance_matrix
        self.centroids = None
        self.labels_ = None
        self.random_state = np.random.RandomState(random_state)

    def fit(self, X):
        if self.distance_matrix is None:
            self._fit_euclidean(X)
        else:
            self._fit_custom_distance(X)

        return self

    def _fit_euclidean(self, X):
        n_samples, n_features = X.shape
        random_indices = self.random_state.choice(n_samples, self.n_clusters, replace=False)
        self.centroids = X[random_indices]

        for _ in range(self.max_iter):
            distances = self._compute_distances(X)
            self.labels_ = np.argmin(distances, axis=1)
            new_centroids = np.array([X[self.labels_ == i].mean(axis=0) for i in range(self.n_clusters)])

            if np.all(np.abs(new_centroids - self.centroids) < self.tol):
                break
            self.centroids = new_centroids

    def _fit_custom_distance(self, X):
        n_samples = X.shape[0]
        for _ in range(self.max_iter):
            distances = self.distance_matrix
            self.labels_ = np.argmin(distances, axis=1)
            new_centroids = np.array([X[self.labels_ == i].mean(axis=0) for i in range(self.n_clusters)])

            if np.all(np.abs(new_centroids - self.centroids) < self.tol):
                break
            self.centroids = new_centroids

    def _compute_distances(self, X):
        return np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)

    def predict(self, X):
        if self.centroids is None:
            raise RuntimeError("You must fit the model before predicting.")
        distances = self._compute_distances(X)
        return np.argmin(distances, axis=1)

    def set_distance_matrix(self, distance_matrix):
        self.distance_matrix = distance_matrix