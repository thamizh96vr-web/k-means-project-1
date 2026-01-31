
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans as SKKMeans

np.random.seed(42)

X, y_true = make_blobs(
    n_samples=400,
    centers=3,
    cluster_std=1.2,
    random_state=42
)


class KMeansScratch:

    def __init__(self, k=3, max_iter=100, tol=1e-4):
        self.k = k
        self.max_iter = max_iter
        self.tol = tol

    def _distance(self, X, centroids):
        return np.sqrt(((X[:, np.newaxis] - centroids) ** 2).sum(axis=2))

    def _initialize_centroids(self, X):
        random_idx = np.random.choice(len(X), self.k, replace=False)
        return X[random_idx]

    def _assign_clusters(self, distances):
        return np.argmin(distances, axis=1)

    def _update_centroids(self, X, labels):
        new_centroids = np.array([
            X[labels == i].mean(axis=0)
            for i in range(self.k)
        ])
        return new_centroids

    def _compute_wcss(self, X, labels, centroids):
        wcss = 0
        for i in range(self.k):
            cluster_points = X[labels == i]
            wcss += ((cluster_points - centroids[i]) ** 2).sum()
        return wcss

    def fit(self, X):
        self.centroids = self._initialize_centroids(X)

        for _ in range(self.max_iter):
            distances = self._distance(X, self.centroids)
            labels = self._assign_clusters(distances)
            new_centroids = self._update_centroids(X, labels)

            shift = np.linalg.norm(self.centroids - new_centroids)
            self.centroids = new_centroids

            if shift < self.tol:
                break

        self.labels_ = labels
        self.wcss_ = self._compute_wcss(X, labels, self.centroids)
        return self

    def predict(self, X):
        distances = self._distance(X, self.centroids)
        return self._assign_clusters(distances)


custom_model = KMeansScratch(k=3)
custom_model.fit(X)

print("\n===== CUSTOM KMEANS =====")
print("WCSS:", custom_model.wcss_)
print("First 20 cluster assignments:")
print(custom_model.labels_[:20])


sk_model = SKKMeans(n_clusters=3, random_state=42)
sk_model.fit(X)

print("\n===== SKLEARN KMEANS =====")
print("WCSS:", sk_model.inertia_)
print("First 20 cluster assignments:")
print(sk_model.labels_[:20])


plt.figure(figsize=(8,6))

plt.scatter(
    X[:, 0],
    X[:, 1],
    c=custom_model.labels_,
    s=40
)

plt.scatter(
    custom_model.centroids[:, 0],
    custom_model.centroids[:, 1],
    marker="X",
    s=250
)

plt.title("K-Means Clustering (From Scratch)")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")

plt.show()





