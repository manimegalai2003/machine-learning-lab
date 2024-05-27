pip install numpy pandas matplotlib scikit-learn scipy
import numpy as np
import pandas as pd
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
    for _ in range(max_iters):
        densities = np.array([pi[i] * multivariate_normal.pdf(X, mean=means[i], cov=covs[i]) for i in range(k)]).T
        responsibilities = densities / densities.sum(axis=1, keepdims=True)
        new_means = np.array([responsibilities[:, i].dot(X) / responsibilities[:, i].sum() for i in range(k)])
        new_covs = np.array([np.dot((responsibilities[:, i] * (X - new_means[i])).T, X - new_means[i]) / responsibilities[:, i].sum() for i in range(k)])
        new_pi = responsibilities.mean(axis=0)
        if np.allclose(new_means, means) and np.allclose(new_covs, covs) and np.allclose(new_pi, pi):
            break
        means, covs, pi = new_means, new_covs, new_pi
    return means, responsibilities.argmax(axis=1)

means, predY_gmm = gaussian_mixture(X[:, 2:], 3)
plt.subplot(1, 3, 3)
plt.scatter(X[:, 2], X[:, 3], c=colormap[predY_gmm], s=40)
plt.scatter(means[:, 0], means[:, 1], marker='*', s=200, c='orange')
plt.title('GMM Classification')

plt.tight_layout()
plt.show()
