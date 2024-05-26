import streamlit as st
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split  # Add this import

# Load the Iris dataset
def load_iris_dataset():
    iris_data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',
                            header=None)
    X = iris_data.iloc[:, :-1].values
    y = iris_data.iloc[:, -1].values
    return X, y

# Custom KNN Classifier
class KNNClassifier:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def predict(self, X_test):
        y_pred = [self._predict(x) for x in X_test]
        return np.array(y_pred)

    def _predict(self, x):
        distances = [self.euclidean_distance(x, x_train) for x_train in self.X_train]
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

# Streamlit App
def main():
    st.title('KNN Classifier for Iris Dataset')

    # Load data
    X, y = load_iris_dataset()

    # Sidebar options
    k_value = st.sidebar.slider('Select K value:', 1, 10, 3)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Model training and prediction
    knn_classifier = KNNClassifier(k=k_value)
    knn_classifier.fit(X_train, y_train)
    y_pred = knn_classifier.predict(X_test)

    # Evaluation
    accuracy = np.mean(y_pred == y_test)

    # Display results
    st.write(f'Accuracy: {accuracy:.2f}')
    st.write('Predicted labels:', y_pred)

if __name__ == '__main__':
    main()
