import streamlit as st
import joblib
import pandas as pd
import numpy as np

# ------------------ Load Model & Preprocessors ------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("star_classifier_model.pkl")
    scaler = joblib.load("scaler.pkl")
    color_encoder = joblib.load("color_encoder.pkl")
    spectral_encoder = joblib.load("spectral_encoder.pkl")
    return model, scaler, color_encoder, spectral_encoder

model, scaler, color_encoder, spectral_encoder = load_artifacts()

# ------------------ Star Type Mapping ------------------
star_type_dict = {
    0: "Brown Dwarf ⭐",
    1: "Red Dwarf 🔴",
    2: "White Dwarf ⚪",
    3: "Main Sequence ☀️",
    4: "Supergiant 🌟",
    5: "Hypergiant 💫"
}

# ------------------ Universe Background + Styling ------------------
st.markdown(
    """
    <style>
    body {
        background: url('https://www.pexels.com/search/space%20background/?utm_source');
        background-size: cover;
        background-attachment: fixed;
        color: white;
        font-family: 'Trebuchet MS', sans-serif;
    }
    h1 {
        font-size: 50px;
        text-align: center;
        background: linear-gradient(90deg, #ff8c00, #e52e71, #00c6ff, #adff2f);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: bold;
    }
    h2, h3 {
        color: #ffd700;
    }
    .stButton>button {
        background: linear-gradient(45deg, #6a11cb, #2575fc);
        color: white;
        border-radius: 12px;
        padding: 10px 20px;
        font-size: 18px;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("<h1>🌌 Star Classification App 🌌</h1>", unsafe_allow_html=True)

# ------------------ User Input ------------------
st.markdown("### Enter Star Properties 🌠")

temperature = st.number_input("Surface Temperature (K)", min_value=1000, max_value=50000, value=5778)
luminosity = st.number_input("Luminosity (L/Lo)", min_value=0.0001, max_value=100000.0, value=1.0)
radius = st.number_input("Radius (R/Ro)", min_value=0.01, max_value=1000.0, value=1.0)
abs_magnitude = st.number_input("Absolute Magnitude", min_value=-10.0, max_value=20.0, value=4.83)

color = st.selectbox("Star Color", options=color_encoder.classes_)
spectral = st.selectbox("Spectral Class", options=spectral_encoder.classes_)

# ------------------ Prediction ------------------
def predict_star(temp, lum, rad, mag, color, spectral):
    try:
        # Encode categorical features
        color_encoded = color_encoder.transform([color])[0]
        spectral_encoded = spectral_encoder.transform([spectral])[0]

        # Prepare input
        input_data = pd.DataFrame([[temp, lum, rad, mag, color_encoded, spectral_encoded]],
                                  columns=["Temperature", "L", "R", "A_M", "Color", "Spectral_Class"])

        # Scale numeric features
        numeric_cols = ["Temperature", "L", "R", "A_M"]
        input_data[numeric_cols] = scaler.transform(input_data[numeric_cols])

        # Predict
        prediction = model.predict(input_data)[0]
        return star_type_dict.get(prediction, "Unknown Type ❓")

    except Exception as e:
        return f"⚠️ Error: {str(e)}"

if st.button("🔭 Classify Star"):
    result = predict_star(temperature, luminosity, radius, abs_magnitude, color, spectral)
    st.subheader("✨ Prediction Result ✨")
    st.success(f"The star is classified as: **{result}**")
