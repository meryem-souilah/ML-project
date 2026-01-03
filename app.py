import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

# Define the OutlierRemover class
class OutlierRemover(BaseEstimator, TransformerMixin):
    def __init__(self, method='median', threshold=1.5):
        self.method = method
        self.threshold = threshold  # IQR threshold

    def fit(self, X, y=None):
        self.Q1 = X.quantile(0.25)
        self.Q3 = X.quantile(0.75)
        self.IQR = self.Q3 - self.Q1
        return self

    def transform(self, X):
        lower_bound = self.Q1 - self.threshold * self.IQR
        upper_bound = self.Q3 + self.threshold * self.IQR
        X_no_outliers = X.copy()

        for col in X.select_dtypes(include=[np.number]).columns:
            if self.method == 'median':
                median = X[col].median()
                X_no_outliers[col] = np.where(
                    (X[col] < lower_bound[col]) | (X[col] > upper_bound[col]),
                    median,
                    X[col]
                )
            elif self.method == 'clip':
                X_no_outliers[col] = X_no_outliers[col].clip(lower=lower_bound[col], upper=upper_bound[col])

        return X_no_outliers

import pandas as pd
import streamlit as st
import joblib
import json

# Load the model with joblib
model = joblib.load('final_model.pkl')

# Load the schema for expected columns
with open("input_schema.json", "r") as f:
    schema = json.load(f)

expected_columns = schema["columns"]

st.title("IoT Intrusion Detection System 🚀")
st.write("Upload a CSV file containing network flow features for Attack Type prediction")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        # Try reading the CSV with a different encoding
        df = pd.read_csv(uploaded_file, encoding='ISO-8859-1')  # You can also try 'windows-1252'

        # Show a preview of the uploaded data
        st.subheader("Uploaded Data Preview")
        st.write(df.head())

    except Exception as e:
        st.error(f"Error reading the CSV file: {e}")

    # Show a preview of the uploaded data
    st.subheader("Uploaded Data Preview")
    st.write(df.head())

    # Align the columns with the model's expected input schema
    for col in expected_columns:
        if col not in df.columns:
            df[col] = 0  # Add missing columns with default values (0)

    # Ensure the columns are ordered correctly (this step is important)
    df = df[expected_columns]

    # Make predictions with the model
    predictions = model.predict(df)

    # Show predictions
    st.subheader("Predictions")
    st.write(predictions)

    # Add the predicted Attack Type to the dataframe
    df["Predicted_Attack_Type"] = predictions

    # Display the dataframe with predictions
    st.write(df)

    # Option to download the results as CSV
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download predictions as CSV",
        data=csv,
        file_name="predictions.csv",
        mime="text/csv",
    )
