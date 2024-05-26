pip install matplotlib
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

# Generate synthetic data
np.random.seed(0)
cluster1 = np.random.normal(loc=[-2, -2], scale=[1, 1], size=(100, 2))
cluster2 = np.random.normal(loc=[2, 2], scale=[1, 1], size=(100, 2))
data = np.vstack((cluster1, cluster2))

# K-means clustering
def kmeans(data, k, max_iters=100):
    centroids = data[np.random.choice(data.shape[0], k, replace=False)]
    for _ in range(max_iters):
        distances = np.sqrt(np.sum((data[:, np.newaxis] - centroids)**2, axis=2))
        labels = np.argmin(distances, axis=1)
        new_centroids = np.array([data[labels == i].mean(axis=0) for i in range(k)])
        if np.all(new_centroids == centroids):
            break
        centroids = new_centroids
    return centroids, labels

# EM algorithm for Gaussian Mixture Model
def expectation_maximization(data, k, max_iters=100):
    n, m = data.shape
    weights = np.ones(k) / k
    means = data[np.random.choice(n, k, replace=False)]
    covariances = np.array([np.eye(m) for _ in range(k)])

    for _ in range(max_iters):
        likelihoods = np.array([multivariate_normal.pdf(data, mean=means[i], cov=covariances[i]) for i in range(k)])
        responsibilities = (likelihoods * weights) / np.sum(likelihoods * weights, axis=0)
        new_means = np.dot(responsibilities, data) / np.sum(responsibilities, axis=1)[:, np.newaxis]
        new_covariances = np.array([np.dot((responsibilities[i] * (data - new_means[i])).T, data - new_means[i]) / np.sum(responsibilities[i]) for i in range(k)])
        new_weights = np.sum(responsibilities, axis=1) / n

        if np.allclose(new_means, means) and np.allclose(new_covariances, covariances) and np.allclose(new_weights, weights):
            break

        means, covariances, weights = new_means, new_covariances, new_weights

    return means, covariances, weights

# Streamlit app
st.title("Clustering Comparison: EM vs K-means")

# Sidebar options
algorithm = st.sidebar.selectbox("Select Clustering Algorithm", ["K-means", "EM"])
num_clusters = st.sidebar.slider("Number of Clusters", min_value=2, max_value=5)

# Plotting
if algorithm == "K-means":
    centroids, labels = kmeans(data, num_clusters)
    st.write(f"K-means Clustering with {num_clusters} clusters")
elif algorithm == "EM":
    means, covariances, weights = expectation_maximization(data, num_clusters)
    st.write(f"EM Clustering with {num_clusters} clusters")

plt.figure(figsize=(8, 6))
if algorithm == "K-means":
    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', alpha=0.8, edgecolors='k')
    plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', c='r', s=100, label='Centroids')
elif algorithm == "EM":
    for i in range(num_clusters):
        rv = multivariate_normal(means[i], covariances[i])
        x, y = np.meshgrid(np.linspace(-5, 5, 100), np.linspace(-5, 5, 100))
        xy = np.column_stack([x.flat, y.flat])
        z = rv.pdf(xy).reshape(x.shape)
        plt.contour(x, y, z, levels=5)
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
st.pyplot()
