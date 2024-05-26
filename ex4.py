import streamlit as st
import pandas as pd
import numpy as np

class NaiveBayesClassifier:
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.parameters = {}
        
        for i, c in enumerate(self.classes):
            X_c = X[np.where(y == c)]
            self.parameters[c] = {
                'mean': X_c.mean(axis=0),
                'var': X_c.var(axis=0),
                'prior': X_c.shape[0] / X.shape[0]
            }

    def predict(self, X):
        posteriors = []
        for x in X:
            posteriors.append([self._posterior(x, c) for c in self.classes])
        return self.classes[np.argmax(posteriors, axis=1)]
    
    def _posterior(self, x, c):
        mean = self.parameters[c]['mean']
        var = self.parameters[c]['var']
        prior = self.parameters[c]['prior']
        posterior = np.sum(-0.5 * np.log(2. * np.pi * var) - ((x - mean) ** 2) / (2. * var))
        return posterior + np.log(prior)

def main():
    st.title("Tennis Data Classifier")

    # File upload
    uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write("The first 5 rows of data:")
        st.write(data.head())

        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]

        # Convert categorical data to numerical
        for col in X.columns:
            X[col] = X[col].astype('category').cat.codes
        y = y.astype('category').cat.codes

        # Split data into train and test sets
        split_ratio = 0.8
        indices = np.random.permutation(len(X))
        train_size = int(len(X) * split_ratio)
        train_idx, test_idx = indices[:train_size], indices[train_size:]
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # Train classifier
        classifier = NaiveBayesClassifier()
        classifier.fit(X_train.to_numpy(), y_train.to_numpy())

        # Predict and evaluate
        y_pred = classifier.predict(X_test.to_numpy())
        accuracy = np.mean(y_pred == y_test.to_numpy())

        st.write(f"Accuracy: {accuracy:.2f}")

if _name_ == "_main_":
    main()
