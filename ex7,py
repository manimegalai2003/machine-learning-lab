import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

def generate_data():
    np.random.seed(0)
    # Generate two clusters of data
    cluster1 = np.random.normal(0, 1, size=(100, 2))
    cluster2 = np.random.normal(3, 1, size=(100, 2))
    data = np.vstack((cluster1, cluster2))
    return data

def k_means(data, k, max_iter=100):
    centroids = data[np.random.choice(len(data), k, replace=False)]
    for _ in range(max_iter):
        # Assign points to clusters based on closest centroid
        labels = np.argmin(np.linalg.norm(data[:, None] - centroids, axis=2), axis=1)
        # Update centroids based on cluster means
        centroids = np.array([data[labels == i].mean(axis=0) for i in range(k)])
    return labels, centroids

def em_algorithm(data, k, max_iter=100):
    np.random.seed(0)
    # Initialize parameters randomly
    weights = np.random.dirichlet(np.ones(k), size=1).flatten()
    means = np.random.uniform(-1, 4, size=(k, data.shape[1]))
    covariances = np.array([np.eye(data.shape[1]) for _ in range(k)])
    log_likelihoods = []
    for _ in range(max_iter):
        # E-step: Calculate responsibilities
        responsibilities = np.array([weights[i] * multivariate_normal.pdf(data, mean=means[i], cov=covariances[i]) for i in range(k)])
        responsibilities /= responsibilities.sum(axis=0)

        # M-step: Update parameters
        Nk = responsibilities.sum(axis=1)
        weights = Nk / len(data)
        means = responsibilities @ data / Nk[:, None]
        covariances = np.array([np.diag((responsibilities[i][:, None] * (data - means[i])).T @ (data - means[i]) / Nk[i]) for i in range(k)])

        # Calculate log-likelihood
        log_likelihood = np.sum(np.log(np.sum([weights[i] * multivariate_normal.pdf(data, mean=means[i], cov=covariances[i]) for i in range(k)], axis=0)))
        log_likelihoods.append(log_likelihood)

    labels = np.argmax(responsibilities, axis=0)
    return labels, means

# Generate data
data = generate_data()

# Sidebar
st.sidebar.title("Clustering Algorithms")
algorithm = st.sidebar.radio("Select Algorithm", ("K-means", "Expectation-Maximization (EM)"))

# Main content
st.title("Comparison of Clustering Algorithms")
st.header("Generated Data")
st.write("Data Points:", len(data))

if algorithm == "K-means":
    k = st.slider("Number of Clusters (K)", min_value=2, max_value=5, value=2)
    labels, centroids = k_means(data, k)
    st.subheader("K-means Clustering")
elif algorithm == "Expectation-Maximization (EM)":
    k = st.slider("Number of Clusters (K)", min_value=2, max_value=5, value=2)
    labels, means = em_algorithm(data, k)
    st.subheader("Expectation-Maximization (EM) Clustering")

# Visualize clusters
fig, ax = plt.subplots()
scatter = ax.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis')
legend = ax.legend(*scatter.legend_elements(), loc="lower right", title="Clusters")
ax.add_artist(legend)
ax.set_xlabel("X")
ax.set_ylabel("Y")
st.pyplot(fig)
