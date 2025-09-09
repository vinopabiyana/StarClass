import streamlit as st
import pandas as pd
import joblib
import base64

# -----------------------
# Load Models & Preprocessing Objects
# -----------------------
@st.cache_resource
def load_models():
    model = joblib.load("star_classifier_model.pkl")
    scaler = joblib.load("scaler.pkl")
    # Skip Color and Spectral_Class if they cause errors
    # color_encoder = joblib.load("color_encoder.pkl")
    # spectral_encoder = joblib.load("spectral_encoder.pkl")
    return model, scaler

model, scaler = load_models()

# Numeric columns
numeric_cols = ["Temperature", "L", "R", "A_M"]

# -----------------------
# Page Configuration
# -----------------------
st.set_page_config(
    page_title="⭐ Star Type Classification",
    page_icon="🌌",
    layout="wide",
)

# -----------------------
# Universe Background
# -----------------------
def set_universe_background():
    page_bg_img = '''
    <style>
    body {
    background-image: url("https://images.unsplash.com/photo-1581090700227-cd71f1f0f79b?fit=crop&w=1920&q=80");
    background-size: cover;
    background-attachment: fixed;
    }
    </style>
    '''
    st.markdown(page_bg_img, unsafe_allow_html=True)

set_universe_background()

# -----------------------
# Title
# -----------------------
st.markdown("<h1 style='text-align:center; color:#FFD700;'>⭐ Star Type Classification 🌌</h1>", unsafe_allow_html=True)
st.markdown("---")

# -----------------------
# Single Star Prediction
# -----------------------
st.subheader("🌟 Predict a Single Star Type")
with st.form("single_star_form"):
    Temperature = st.number_input("Temperature", min_value=100, max_value=50000, value=5000)
    L = st.number_input("Luminosity (L)", min_value=0.0001, value=1.0)
    R = st.number_input("Radius (R)", min_value=0.01, value=1.0)
    A_M = st.number_input("Absolute Magnitude (A_M)", value=5.0)
    submitted = st.form_submit_button("Predict Star Type")

    if submitted:
        input_df = pd.DataFrame([[Temperature, L, R, A_M]], columns=numeric_cols)
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)[0]

        # Map number to star type
        star_type_mapping = {
            0: "Red Dwarf",
            1: "Brown Dwarf",
            2: "White Dwarf",
            3: "Main Sequence",
            4: "Supergiant",
            5: "Hypergiant"
        }
        st.success(f"Predicted Star Type: ⭐ {star_type_mapping[prediction]} ⭐")

st.markdown("---")

# -----------------------
# Batch Prediction from CSV
# -----------------------
st.subheader("🌌 Batch Star Prediction from CSV")

uploaded_file = st.file_uploader("Upload CSV for batch prediction", type=["csv"])
if uploaded_file:
    try:
        batch_df = pd.read_csv(uploaded_file)

        # Preprocess numeric columns
        batch_df[numeric_cols] = scaler.transform(batch_df[numeric_cols])

        # Predict star type
        batch_predictions = model.predict(batch_df[numeric_cols])
        batch_df["Predicted Type"] = [star_type_mapping[i] for i in batch_predictions]

        st.success("Batch prediction completed!")
        st.dataframe(batch_df)

        # Download CSV
        csv = batch_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Predictions as CSV",
            data=csv,
            file_name="batch_star_predictions.csv",
            mime="text/csv",
        )

    except Exception as e:
        st.error(f"Error in batch prediction: {e}")

# -----------------------
# Footer
# -----------------------
st.markdown("---")
st.markdown("<p style='text-align:center; color:#FFFFFF;'>Developed with ❤️ by Vino Pabiyana</p>", unsafe_allow_html=True)
