import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ==========================
# 🎨 Page Config
# ==========================
st.set_page_config(
    page_title="Star Type Classifier 🌌",
    page_icon="✨",
    layout="wide"
)

# ==========================
# 🌌 Background Universe Effect (CSS)
# ==========================
st.markdown(
    """
    <style>
    body {
        background: radial-gradient(ellipse at bottom, #1b2735 0%, #090a0f 100%);
        color: white;
    }
    .star {
        position: absolute;
        width: 2px;
        height: 2px;
        background: white;
        animation: fall 4s linear infinite;
    }
    @keyframes fall {
        0% { transform: translateY(-10px); opacity: 1; }
        100% { transform: translateY(100vh); opacity: 0; }
    }
    </style>
    <script>
    for (let i = 0; i < 50; i++) {
        let star = document.createElement("div");
        star.className = "star";
        star.style.left = Math.random() * 100 + "vw";
        star.style.animationDuration = (Math.random() * 3 + 2) + "s";
        star.style.opacity = Math.random();
        document.body.appendChild(star);
    }
    </script>
    """,
    unsafe_allow_html=True
)

# ==========================
# 📦 Load Models Safely
# ==========================
@st.cache_resource
def load_models():
    required_files = {
        "model": "star_classifier_model.pkl",  # ✅ Use your saved file
        "scaler": "scaler.pkl",
        "color_encoder": "color_encoder.pkl",
        "spectral_encoder": "spectral_encoder.pkl"
    }

    loaded = {}
    missing_files = []

    for key, filename in required_files.items():
        if os.path.exists(filename):
            loaded[key] = joblib.load(filename)
        else:
            missing_files.append(filename)
            loaded[key] = None

    return loaded, missing_files


models, missing = load_models()

if missing:
    st.error(f"❌ Missing files: {', '.join(missing)}. Please upload them to run predictions.")
else:
    st.success("✅ All models loaded successfully!")

# ==========================
# 🔮 Star Type Mapping
# ==========================
star_type_map = {
    0: "Brown Dwarf",
    1: "Red Dwarf",
    2: "White Dwarf",
    3: "Main Sequence",
    4: "Supergiant",
    5: "Hypergiant"
}

# ==========================
# 📊 Prediction Function
# ==========================
def predict_star(input_df):
    model = models["model"]
    scaler = models["scaler"]
    color_encoder = models["color_encoder"]
    spectral_encoder = models["spectral_encoder"]

    # Encode categorical
    input_df["Color"] = color_encoder.transform(input_df["Color"])
    input_df["Spectral_Class"] = spectral_encoder.transform(input_df["Spectral_Class"])

    # Scale numeric
    scaled_data = scaler.transform(input_df)

    # Predict
    prediction = model.predict(scaled_data)
    return [star_type_map[p] for p in prediction]

# ==========================
# 🌟 Streamlit App Layout
# ==========================
st.title("🌌 Star Type Classification App")
st.markdown("Upload data or enter values to classify stars into their types.")

tab1, tab2 = st.tabs(["🔹 Single Prediction", "📂 Batch Prediction"])

# --- Single Prediction ---
with tab1:
    st.subheader("Enter Star Features:")

    temperature = st.number_input("Temperature (K)", min_value=1000, max_value=40000, value=5778)
    luminosity = st.number_input("Luminosity (L/Lo)", min_value=0.0001, value=1.0, format="%.4f")
    radius = st.number_input("Radius (R/Ro)", min_value=0.1, value=1.0)
    absolute_magnitude = st.number_input("Absolute Magnitude (Mv)", min_value=-10.0, max_value=20.0, value=4.83)
    color = st.selectbox("Color", ["Red", "Blue", "White", "Yellow", "Orange"])
    spectral_class = st.selectbox("Spectral Class", ["O", "B", "A", "F", "G", "K", "M"])

    if st.button("🔮 Predict Star Type"):
        input_df = pd.DataFrame([{
            "Temperature": temperature,
            "L": luminosity,
            "R": radius,
            "A_M": absolute_magnitude,
            "Color": [color][0:1],  # as series
            "Spectral_Class": [spectral_class][0:1]
        }])
        result = predict_star(input_df)
        st.success(f"🌟 Predicted Star Type: **{result[0]}**")

# --- Batch Prediction ---
with tab2:
    st.subheader("Upload CSV File for Batch Prediction")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file:
        batch_df = pd.read_csv(uploaded_file)
        st.write("📂 Uploaded Data Preview:", batch_df.head())

        try:
            results = predict_star(batch_df)
            batch_df["Predicted_Star_Type"] = results
            st.success("✅ Batch Prediction Complete")
            st.dataframe(batch_df)

            # Download results
            csv = batch_df.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download Predictions", csv, "predictions.csv", "text/csv")
        except Exception as e:
            st.error(f"⚠️ Error during prediction: {e}")
