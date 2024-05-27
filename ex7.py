import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from sklearn.datasets import load_iris

# Load Iris dataset manually
iris = load_iris()
X = iris.data
y = iris.target

plt.figure(figsize=(14, 7))
colormap = np.array(['red', 'lime', 'black'])

# REAL PLOT
plt.subplot(1, 3, 1)
plt.scatter(X[:, 2], X[:, 3], c=colormap[y], s=40)
plt.title('Real')

# K-PLOT (K-Means without sklearn)
def k_means(X, k, max_iters=100):
    centroids = X[np.random.choice(X.shape[0], k, replace=False)]
    for _ in range(max_iters):
        distances = np.linalg.norm(X[:, None] - centroids, axis=-1)
        labels = np.argmin(distances, axis=1)
        new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(k)])
        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids
    return centroids, labels

centroids, predY_kmeans = k_means(X[:, 2:], 3)
plt.subplot(1, 3, 2)
plt.scatter(X[:, 2], X[:, 3], c=colormap[predY_kmeans], s=40)
plt.scatter(centroids[:, 0], centroids[:, 1], marker='*', s=200, c='orange')
plt.title('K-Means')

# GMM PLOT (Gaussian Mixture Model without sklearn)
def gaussian_mixture(X, k, max_iters=100):
    n, d = X.shape
    pi = np.ones(k) / k
    means = X[np.random.choice(n, k, replace=False)]
    covs = np.array([np.eye(d)] * k)
