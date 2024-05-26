import pandas as pd
import numpy as np
from collections import Counter
import streamlit as st

# File uploader
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Load data
    msg = pd.read_csv(uploaded_file, names=['message', 'label'])
    st.write("Total Instances of Dataset: ", msg.shape[0])
    msg['labelnum'] = msg.label.map({'pos': 1, 'neg': 0})

    X = msg.message
    y = msg.labelnum

    # Split data (using numpy)
    def train_test_split(X, y, test_size=0.25, random_state=None):
        if random_state:
            np.random.seed(random_state)
        indices = np.arange(X.shape[0])
        np.random.shuffle(indices)
        split_idx = int(X.shape[0] * (1 - test_size))
        train_idx, test_idx = indices[:split_idx], indices[split_idx:]
        return X.iloc[train_idx], X.iloc[test_idx], y.iloc[train_idx], y.iloc[test_idx]

    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y)

    if len(Xtrain) == 0 or len(Xtest) == 0:
        st.write("Error: Insufficient data for training and testing.")
    else:
        # Count Vectorization
        def count_vectorize(corpus, vocab=None):
            if vocab is None:
                vocab = Counter()
                for doc in corpus:
                    vocab.update(doc.split())
                vocab = sorted(vocab.keys())

            def vectorize(doc):
                vec = np.zeros(len(vocab))
                word_counts = Counter(doc.split())
                for i, word in enumerate(vocab):
                    vec[i] = word_counts[word]
                return vec

            return np.array([vectorize(doc) for doc in corpus]), vocab

        Xtrain_dm, vocab = count_vectorize(Xtrain)
        Xtest_dm, _ = count_vectorize(Xtest, vocab)

        # Custom Multinomial Naive Bayes
        class MultinomialNB:
            def fit(self, X, y):
                self.classes = np.unique(y)
                self.class_count = np.array([np.sum(y == c) for c in self.classes])
                self.feature_count = np.array([np.sum(X[y == c], axis=0) for c in self.classes])
                self.feature_log_prob = np.log((self.feature_count + 1) / (self.feature_count.sum(axis=1, keepdims=True) + X.shape[1]))
                self.class_log_prior = np.log(self.class_count / y.size)

            def predict(self, X):
                jll = X @ self.feature_log_prob.T + self.class_log_prior
                return self.classes[np.argmax(jll, axis=1)]

        clf = MultinomialNB()
        clf.fit(Xtrain_dm, ytrain)
        pred = clf.predict(Xtest_dm)

        # Accuracy Metrics
        accuracy = np.mean(ytest == pred)
        recall = np.diag(np.bincount(ytest * 2 + pred)) / np.bincount(ytest * 2 + pred).sum(axis=1)
        precision = np.diag(np.bincount(ytest * 2 + pred)) / np.bincount(ytest * 2 + pred).sum(axis=0)
        confusion_matrix = np.array([[np.sum((ytest == i) & (pred == j)) for j in np.unique(pred)] for i in np.unique(ytest)])

        # Display results
        st.write("Accuracy: {:.2f}".format(accuracy))
        st.write("Recall: {:.2f}".format(recall.mean()))
        st.write("Precision: {:.2f}".format(precision.mean()))
        st.write("Confusion Matrix:")
        st.write(pd.DataFrame(confusion_matrix, columns=['neg', 'pos'], index=['neg', 'pos']))
else:
    st.write("Please upload a CSV file.")
