import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from scipy.stats import multivariate_normal

# Load Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

colormap = np.array(['red', 'lime', 'black'])

# K-Means Algorithm
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

# EM Algorithm (Gaussian Mixture Model)
def gaussian_mixture(X, k, max_iters=100):
    n, d = X.shape
    pi = np.ones(k) / k
    means = X[np.random.choice(n, k, replace=False)]
    covs = np.array([np.eye(d)] * k)
    for _ in range(max_iters):
        densities = np.array([pi[i] * multivariate_normal.pdf(X, mean=means[i], cov=covs[i]) for i in range(k)]).T
        responsibilities = densities / densities.sum(axis=1, keepdims=True)
        new_means = np.array([responsibilities[:, i].dot(X) / responsibilities[:, i].sum() for i in range(k)])
        new_covs = np.array([responsibilities[:, i].dot((X - new_means[i]).T.dot(X - new_means[i])) / responsibilities[:, i].sum() for i in range(k)])
        new_pi = responsibilities.mean(axis=0)
        if np.allclose(new_means, means) and np.allclose(new_covs, covs) and np.allclose(new_pi, pi):
            break
        means, covs, pi = new_means, new_covs, new_pi
    return means, responsibilities.argmax(axis=1)

# Streamlit App Layout
st.title("Clustering Comparison: K-Means vs EM Algorithm")
st.write("## Iris Dataset Clustering")

# K-Means Clustering
centroids_kmeans, labels_kmeans = k_means(X[:, 2:], 3)

# EM Algorithm Clustering
means_gmm, labels_gmm = gaussian_mixture(X[:, 2:], 3)

# Plotting
fig, axs = plt.subplots(1, 3, figsize=(21, 7))

# Real Plot
axs[0].scatter(X[:, 2], X[:, 3], c=colormap[y], s=40)
axs[0].set_title('Real')

# K-Means Plot
axs[1].scatter(X[:, 2], X[:, 3], c=colormap[labels_kmeans], s=40)
axs[1].scatter(centroids_kmeans[:, 0], centroids_kmeans[:, 1], marker='*', s=200, c='orange')
axs[1].set_title('K-Means')

# GMM Plot
axs[2].scatter(X[:, 2], X[:, 3], c=colormap[labels_gmm], s=40)
axs[2].scatter(means_gmm[:, 0], means_gmm[:, 1], marker='*', s=200, c='orange')
axs[2].set_title('GMM Classification')

plt.tight_layout()
st.pyplot(fig)
