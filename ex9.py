import streamlit as st
import numpy as np

def lwr(X, y, query_point, tau=0.1):
    # Locally Weighted Regression implementation (same as previous code)

    # Example data
    X = np.array([[1, 2], [3, 4], [5, 6]])  # Features
    y = np.array([3, 5, 7])  # Target values

    # Query point(s) for prediction
    query_points = np.array([2, 4])
    predictions = [lwr(X, y, q) for q in query_points]
    st.write(predictions)

def main():
    st.title("Locally Weighted Regression")

    # User inputs
    query_point = st.number_input("Enter query point:", min_value=0.0, step=0.1)
    tau = st.slider("Bandwidth (tau):", min_value=0.01, max_value=1.0, value=0.1, step=0.01)

    # Button to trigger prediction
    if st.button("Predict"):
        lwr(X, y, query_point, tau)

if __name__ == "__main__":
    main()
