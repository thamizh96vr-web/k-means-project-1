
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



# ============================================================
# K-MEANS CLUSTERING PROJECT – FULL EXPLANATION
# ============================================================

# In this project, I implemented the K-Means clustering algorithm
# from scratch and compared its behavior with the K-Means
# implementation provided by the scikit-learn library.

# The goal of the project was to understand how K-Means works
# internally and to observe how implementation choices affect
# the final clustering results.

# ------------------------------------------------------------
# DATASET DESCRIPTION
# ------------------------------------------------------------
# The dataset was generated using the make_blobs function.
# It contains 400 data points with two features each.
# The data is grouped around 3 centers, which makes it
# suitable for testing a clustering algorithm.

# A fixed random seed was used to ensure reproducibility,
# meaning the same data is generated each time the code runs.

# ------------------------------------------------------------
# OBJECTIVE OF K-MEANS
# ------------------------------------------------------------
# The main objective of K-Means is to group similar data points
# into K clusters by minimizing the Within-Cluster Sum of Squares
# (WCSS).

# WCSS measures how close data points are to their assigned
# cluster centroids. Lower WCSS values indicate tighter and
# more compact clusters.

# ------------------------------------------------------------
# HOW THE CUSTOM K-MEANS IMPLEMENTATION WORKS
# ------------------------------------------------------------
# Step 1: Centroid Initialization
# In the custom implementation, the initial centroids are
# selected randomly from the dataset.

# Step 2: Distance Calculation
# The Euclidean distance is computed between each data point
# and each centroid to measure similarity.

# Step 3: Cluster Assignment
# Each data point is assigned to the nearest centroid based
# on the smallest distance.

# Step 4: Centroid Update
# New centroids are calculated as the mean of all points
# assigned to each cluster.

# Step 5: Iteration and Convergence
# These steps are repeated until the centroids stop changing
# significantly or the maximum number of iterations is reached.

# ------------------------------------------------------------
# OBSERVATIONS FROM CUSTOM IMPLEMENTATION
# ------------------------------------------------------------
# During execution, the algorithm converged quickly and formed
# three visually distinct clusters.

# However, the final WCSS value obtained from the custom model
# was slightly higher compared to scikit-learn’s K-Means.

# This difference was observed even though both models were
# trained on the same dataset with the same number of clusters.

# ------------------------------------------------------------
# COMPARISON WITH SCIKIT-LEARN K-MEANS
# ------------------------------------------------------------
# Scikit-learn’s K-Means uses K-Means++ initialization by default.
# This method selects initial centroids that are far apart,
# which usually leads to better clustering results.

# In contrast, random initialization in the custom model can
# place centroids close to each other or in less dense regions.
# This can cause the algorithm to converge to a less optimal
# solution with higher WCSS.

# This explains why scikit-learn achieved a lower WCSS value
# compared to the custom implementation.

# ------------------------------------------------------------
# INTERPRETATION OF THE SCATTER PLOT
# ------------------------------------------------------------
# The scatter plot produced by the custom model shows that
# the data points are grouped into three clear clusters.

# The centroids are located near the center of each cluster,
# which confirms that the algorithm is working correctly.

# Although the clustering is visually good, the clusters
# are slightly less compact than what is expected from
# scikit-learn’s result, which matches the WCSS comparison.

# ------------------------------------------------------------
# LIMITATIONS IDENTIFIED
# ------------------------------------------------------------
# The main limitation of this implementation is the use of
# random centroid initialization.

# Another limitation is that K-Means assumes clusters are
# spherical and similar in size, which may not hold for
# all real-world datasets.

# ------------------------------------------------------------
# CONCLUSION
# ------------------------------------------------------------
# This project successfully demonstrates a correct from-scratch
# implementation of the K-Means clustering algorithm.

# The comparison with scikit-learn highlights the importance
# of centroid initialization and shows how K-Means++ can
# improve clustering quality in practice.

# Future improvements could include implementing K-Means++
# initialization to reduce WCSS and improve robustness.




