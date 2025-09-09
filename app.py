import streamlit as st
import joblib
import pandas as pd

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="⭐ Star Type Classification",
    page_icon="🌌",
    layout="wide"
)

# ------------------ BACKGROUND STYLE ------------------
st.markdown(
    """
    <style>
    body {
        background: radial-gradient(ellipse at bottom, #1b2735 0%, #090a0f 100%) !important;
        color: white !important;
    }
    .main {
        background: transparent !important;
    }
    h1, h2, h3, h4, h5, h6, p {
        color: white !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ------------------ LOAD MODELS ------------------
@st.cache_resource
def load_models():
    model = joblib.load("star_classifier_model.pkl")
    scaler = joblib.load("scaler.pkl")
    color_encoder = joblib.load("color_encoder.pkl")
    spectral_encoder = joblib.load("spectral_encoder.pkl")
    return model, scaler, color_encoder, spectral_encoder


model, scaler, color_encoder, spectral_encoder = load_models()

# ------------------ STAR TYPE MAPPING ------------------
star_types = {
    0: "Brown Dwarf",
    1: "Red Dwarf",
    2: "White Dwarf",
    3: "Main Sequence",
    4: "Supergiant",
    5: "Hypergiant"
}

# ------------------ PREDICTION FUNCTION ------------------
def predict_star(input_df):
    # Normalize inputs before encoding
    input_df["Color"] = input_df["Color"].str.strip().str.capitalize()
    input_df["Spectral_Class"] = input_df["Spectral_Class"].str.strip().str.upper()

    # Encode categorical
    input_df["Color"] = color_encoder.transform(input_df["Color"])
    input_df["Spectral_Class"] = spectral_encoder.transform(input_df["Spectral_Class"])

    # Scale numeric
    numeric_cols = ["Temperature", "L", "R", "A_M"]
    input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])

    # Predict
    prediction = model.predict(input_df)[0]
    return star_types[prediction]

# ------------------ APP TITLE ------------------
st.title("🌌 Star Type Classification")
st.markdown("### Predict the type of a star based on its physical properties")

# ------------------ SIDEBAR INPUT ------------------
st.sidebar.header("🔭 Input Star Parameters")

temperature = st.sidebar.number_input("Temperature (K)", min_value=1000, max_value=40000, value=5778)
luminosity = st.sidebar.number_input("Luminosity (L/Lo)", min_value=0.0, max_value=1000000.0, value=1.0)
radius = st.sidebar.number_input("Radius (R/Ro)", min_value=0.0, max_value=1000.0, value=1.0)
absolute_magnitude = st.sidebar.number_input("Absolute Magnitude (M)", min_value=-10.0, max_value=20.0, value=4.83)

color = st.sidebar.selectbox("Color", ["Red", "Blue", "White", "Yellow", "Orange"])
spectral_class = st.sidebar.selectbox("Spectral Class", ["O", "B", "A", "F", "G", "K", "M"])

# ------------------ SINGLE PREDICTION ------------------
if st.sidebar.button("🌟 Predict Star Type"):
    input_df = pd.DataFrame({
        "Temperature": [temperature],
        "L": [luminosity],
        "R": [radius],
        "A_M": [absolute_magnitude],
        "Color": [color],
        "Spectral_Class": [spectral_class]
    })
    result = predict_star(input_df)
    st.success(f"⭐ Predicted Star Type: **{result}**")

# ------------------ BATCH PREDICTION ------------------
st.markdown("## 📂 Batch Prediction (Upload CSV)")
uploaded_file = st.file_uploader("Upload CSV with star data", type=["csv"])

if uploaded_file:
    batch_df = pd.read_csv(uploaded_file)
    try:
        predictions = batch_df.copy()
        predictions["Predicted_Type"] = predict_star(batch_df)
        st.dataframe(predictions)

        # Download link
        csv = predictions.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Predictions",
            data=csv,
            file_name="star_predictions.csv",
            mime="text/csv"
        )
    except Exception as e:
        st.error(f"Error processing file: {e}")
