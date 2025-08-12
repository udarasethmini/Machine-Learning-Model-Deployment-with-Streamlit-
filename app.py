import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import os

# =======================
# Load Model & Scaler
# =======================
@st.cache_resource
def load_model():
    model_path = "model.pkl"
    scaler_path = "scaler.pkl"

    if not os.path.isfile(model_path) or not os.path.isfile(scaler_path):
        st.error(f"❌ Missing model or scaler file!\nMake sure '{model_path}' and '{scaler_path}' are present.")
        return None, None

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

model, scaler = load_model()

# =======================
# Load Dataset
# =======================
@st.cache_data
def load_data():
    csv_path = "data/diabetes.csv"
    if not os.path.isfile(csv_path):
        st.error(f"❌ CSV file not found at path: {csv_path}")
        return pd.DataFrame()
    return pd.read_csv(csv_path)

df = load_data()

# =======================
# Sidebar Navigation
# =======================
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Data Exploration", "Visualisations", "Predict", "Model Performance"])

# =======================
# Pages
# =======================
if page == "Home":
    st.title("Diabetes Prediction App")
    st.markdown("""
    This app predicts the likelihood of diabetes based on patient health data.
    Dataset: **Pima Indians Diabetes Dataset**
    """)

elif page == "Data Exploration":
    st.header("Dataset Overview")
    if df.empty:
        st.warning("Dataset not loaded.")
    else:
        st.write("Shape:", df.shape)
        st.dataframe(df.head())

        if st.checkbox("Show summary statistics"):
            st.write(df.describe())

        if st.checkbox("Show missing values"):
            st.write(df.isnull().sum())

elif page == "Visualisations":
    st.header("Data Visualisations")
    if df.empty:
        st.warning("Dataset not loaded.")
    else:
        col = st.selectbox("Select column for histogram", df.columns[:-1])
        fig = px.histogram(df, x=col, color="Outcome", barmode="overlay")
        st.plotly_chart(fig)

        x_axis = st.selectbox("X-axis", df.columns[:-1])
        y_axis = st.selectbox("Y-axis", df.columns[:-1], index=1)
        fig2 = px.scatter(df, x=x_axis, y=y_axis, color="Outcome", symbol="Outcome")
        st.plotly_chart(fig2)

elif page == "Predict":
    st.header("Make a Prediction")

    if model is None or scaler is None:
        st.warning("⚠ Model and scaler not loaded. Please check files.")
    else:
        pregnancies = st.slider("Pregnancies", 0, 20, 1)
        glucose = st.slider("Glucose", 0, 200, 100)
        bp = st.slider("Blood Pressure", 0, 122, 70)
        skin = st.slider("Skin Thickness", 0, 99, 20)
        insulin = st.slider("Insulin", 0, 846, 79)
        bmi = st.slider("BMI", 0.0, 67.0, 20.0)
        dpf = st.slider("Diabetes Pedigree Function", 0.0, 2.5, 0.5)
        age = st.slider("Age", 21, 81, 33)

        features = np.array([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]])
        scaled_features = scaler.transform(features)

        if st.button("Predict"):
            prediction = model.predict(scaled_features)[0]
            probability = model.predict_proba(scaled_features)[0][prediction]
            if prediction == 1:
                st.error(f"Prediction: Diabetes (Confidence: {probability:.2f})")
            else:
                st.success(f"Prediction: No Diabetes (Confidence: {probability:.2f})")

elif page == "Model Performance":
    st.header("Model Performance")
    if df.empty:
        st.warning("Dataset not loaded.")
    elif model is None or scaler is None:
        st.warning("⚠ Model and scaler not loaded.")
    else:
        X = df.drop("Outcome", axis=1)
        y = df["Outcome"]
        X_scaled = scaler.transform(X)
        preds = model.predict(X_scaled)

        acc = accuracy_score(y, preds)
        st.write(f"Accuracy: **{acc:.4f}**")

        cm = confusion_matrix(y, preds)
        st.write("Confusion Matrix:")
        st.write(cm)

        st.write("Classification Report:")
        report = classification_report(y, preds, output_dict=True)
        st.dataframe(pd.DataFrame(report).transpose())


