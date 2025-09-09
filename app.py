import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time
import matplotlib.pyplot as plt

# ------------------------------
# Load Models & Encoders
# ------------------------------
@st.cache_resource
def load_models():
    model = joblib.load("star_classifier_model.pkl")
    scaler = joblib.load("scaler.pkl")
    color_encoder = joblib.load("color_encoder.pkl")
    spectral_encoder = joblib.load("spectral_encoder.pkl")
    return model, scaler, color_encoder, spectral_encoder

model, scaler, color_encoder, spectral_encoder = load_models()

# ------------------------------
# Define Features
# ------------------------------
numeric_cols = ["Temperature", "L", "R", "A_M"]
categorical_cols = ["Color", "Spectral_Class"]

# ------------------------------
# Safe Encoding for Unseen Categories
# ------------------------------
def safe_encode(encoder, values):
    return [
        val if val in encoder.classes_ else encoder.classes_[0]
        for val in values
    ]

# ------------------------------
# Prediction Function
# ------------------------------
def predict_star(input_df):
    # Safe handling of unseen categories
    input_df["Color"] = safe_encode(color_encoder, input_df["Color"])
    input_df["Spectral_Class"] = safe_encode(spectral_encoder, input_df["Spectral_Class"])

    # Transform with encoders
    input_df["Color"] = color_encoder.transform(input_df["Color"])
    input_df["Spectral_Class"] = spectral_encoder.transform(input_df["Spectral_Class"])

    # Scale numeric columns
    input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])

    # Predictions
    return model.predict(input_df)

# ------------------------------
# Streamlit UI
# ------------------------------
st.set_page_config(page_title="⭐ Star Type Classification", layout="wide")

st.title("⭐ Star Type Classification App")
st.markdown("Predict the **type of a star** based on its characteristics.")

# Sidebar for mode selection
mode = st.sidebar.radio("Choose Mode:", ["Single Prediction", "Batch Prediction (CSV)"])

# ------------------------------
# Single Prediction
# ------------------------------
if mode == "Single Prediction":
    st.header("🔹 Single Star Prediction")

    with st.form("star_form"):
        temp = st.number_input("Temperature (K)", min_value=0, value=5000)
        lum = st.number_input("Luminosity (L/Lo)", min_value=0.0, value=1.0)
        rad = st.number_input("Radius (R/Ro)", min_value=0.0, value=1.0)
        mag = st.number_input("Absolute Magnitude", value=5.0)
        color = st.selectbox("Color", color_encoder.classes_)
        spectral = st.selectbox("Spectral Class", spectral_encoder.classes_)
        submitted = st.form_submit_button("Predict Star Type")

    if submitted:
        input_df = pd.DataFrame([[temp, lum, rad, mag, color, spectral]],
                                columns=numeric_cols + categorical_cols)
        prediction = predict_star(input_df)[0]

        st.success(f"🌟 Predicted Star Type: **{prediction}**")

        # Falling stars effect
        st.balloons()

# ------------------------------
# Batch Prediction
# ------------------------------
else:
    st.header("📂 Batch Star Prediction from CSV")

    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

    if uploaded_file:
        batch_df = pd.read_csv(uploaded_file)

        st.subheader("📄 Uploaded Data Preview")
        st.dataframe(batch_df.head())

        if st.button("Run Batch Prediction"):
            predictions = predict_star(batch_df)
            batch_df["Predicted_Star_Type"] = predictions

            st.subheader("✅ Predictions")
            st.dataframe(batch_df)

            # ------------------------------
            # Visualization
            # ------------------------------
            st.subheader("📊 Prediction Distribution")

            fig, ax = plt.subplots(1, 2, figsize=(12, 5))

            # Bar chart
            pd.Series(predictions).value_counts().plot(kind="bar", ax=ax[0], color="skyblue")
            ax[0].set_title("Count of Star Types")
            ax[0].set_xlabel("Star Type")
            ax[0].set_ylabel("Count")

            # Pie chart
            pd.Series(predictions).value_counts().plot(kind="pie", autopct='%1.1f%%', ax=ax[1])
            ax[1].set_ylabel("")
            ax[1].set_title("Star Type Distribution")

            st.pyplot(fig)

            # Allow download
            st.download_button(
                label="⬇️ Download Predictions",
                data=batch_df.to_csv(index=False),
                file_name="star_predictions.csv",
                mime="text/csv",
            )

