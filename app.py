import streamlit as st
import pandas as pd
import joblib
import numpy as np
import time
import random

# =========================
# Load Models and Encoders
# =========================
@st.cache_resource
def load_models():
    model = joblib.load("star_model.pkl")
    scaler = joblib.load("scaler.pkl")
    color_encoder = joblib.load("color_encoder.pkl")
    spectral_encoder = joblib.load("spectral_encoder.pkl")
    return model, scaler, color_encoder, spectral_encoder

model, scaler, color_encoder, spectral_encoder = load_models()

# Mapping of star type numbers to names
STAR_TYPES = {
    0: "Brown Dwarf",
    1: "Red Dwarf",
    2: "White Dwarf",
    3: "Main Sequence",
    4: "Supergiant",
    5: "Hypergiant"
}

# =========================
# Prediction Function
# =========================
def predict_star(input_df):
    input_df["Color"] = color_encoder.transform(input_df["Color"])
    input_df["Spectral_Class"] = spectral_encoder.transform(input_df["Spectral_Class"])
    scaled_data = scaler.transform(input_df)
    predictions = model.predict(scaled_data)
    return [STAR_TYPES[p] for p in predictions]

# =========================
# Animated Universe Background
# =========================
def set_universe_bg():
    st.markdown(
        """
        <style>
        /* Base space background */
        .stApp {
            background: radial-gradient(ellipse at bottom, #1b2735 0%, #090a0f 100%);
            overflow: hidden;
            color: white;
        }

        /* Twinkling stars */
        .stars {
            width: 1px;
            height: 1px;
            background: transparent;
            box-shadow: 
                1000px 2000px #FFF, 1500px 700px #FFF, 500px 1200px #FFF, 
                700px 300px #FFF, 300px 800px #FFF, 1200px 1500px #FFF;
            animation: animStar 50s linear infinite;
        }

        .stars:after {
            content: " ";
            position: absolute;
            top: 2000px;
            width: 1px;
            height: 1px;
            background: transparent;
            box-shadow: 
                1000px 2000px #FFF, 1500px 700px #FFF, 500px 1200px #FFF, 
                700px 300px #FFF, 300px 800px #FFF, 1200px 1500px #FFF;
        }

        @keyframes animStar {
            from { transform: translateY(0px); }
            to { transform: translateY(-2000px); }
        }

        /* Style improvements */
        .block-container {
            background: rgba(0,0,0,0.6);
            padding: 2rem;
            border-radius: 15px;
        }
        h1, h2, h3 {
            color: #FFD700;
            text-shadow: 2px 2px 5px black;
        }
        </style>
        <div class="stars"></div>
        """,
        unsafe_allow_html=True
    )

set_universe_bg()

# =========================
# Star Animation (emoji fall)
# =========================
def star_animation():
    stars = ["✨", "🌟", "⭐", "💫"]
    for _ in range(15):
        st.markdown(f"<h3 style='text-align:center;'>{random.choice(stars)}</h3>", unsafe_allow_html=True)
        time.sleep(0.1)

# =========================
# Streamlit UI
# =========================
st.title("🌌 Star Type Prediction App")

menu = ["Single Prediction", "Batch Prediction"]
choice = st.sidebar.radio("Choose Mode", menu)

if choice == "Single Prediction":
    st.subheader("🔭 Enter Star Details")

    color = st.selectbox("Star Color", color_encoder.classes_)
    spectral_class = st.selectbox("Spectral Class", spectral_encoder.classes_)
    temperature = st.number_input("Temperature (K)", min_value=1000, max_value=50000, value=5778)
    luminosity = st.number_input("Luminosity (L/Lo)", min_value=0.0001, max_value=100000.0, value=1.0)
    radius = st.number_input("Radius (R/Ro)", min_value=0.1, max_value=1000.0, value=1.0)
    absolute_magnitude = st.number_input("Absolute Magnitude", min_value=-15.0, max_value=20.0, value=4.83)

    if st.button("Predict Star Type"):
        input_data = pd.DataFrame({
            "Temperature": [temperature],
            "Luminosity": [luminosity],
            "Radius": [radius],
            "Absolute_Magnitude": [absolute_magnitude],
            "Color": [color],
            "Spectral_Class": [spectral_class]
        })

        prediction = predict_star(input_data)[0]

        st.success(f"🌟 The predicted star type is: **{prediction}**")
        star_animation()

elif choice == "Batch Prediction":
    st.subheader("📂 Upload CSV File for Batch Prediction")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)
        st.write("📊 Uploaded Data", batch_df)

        try:
            predictions = predict_star(batch_df)
            batch_df["Predicted_Star_Type"] = predictions

            st.success("✅ Batch Prediction Completed!")
            st.write(batch_df)

            # Visualization: Star type distribution
            st.bar_chart(batch_df["Predicted_Star_Type"].value_counts())

        except Exception as e:
            st.error(f"Error in processing: {e}")
