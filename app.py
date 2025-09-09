import streamlit as st
import pandas as pd
import joblib
import base64
import matplotlib.pyplot as plt

# ===============================
# Load Models
# ===============================
@st.cache_resource
def load_models():
    model = joblib.load("star_model.pkl")   # ✅ Correct filename
    scaler = joblib.load("scaler.pkl")
    color_encoder = joblib.load("color_encoder.pkl")
    spectral_encoder = joblib.load("spectral_encoder.pkl")
    return model, scaler, color_encoder, spectral_encoder

model, scaler, color_encoder, spectral_encoder = load_models()

# ===============================
# Star Type Mapping
# ===============================
STAR_TYPES = {
    0: "Brown Dwarf",
    1: "Red Dwarf",
    2: "White Dwarf",
    3: "Main Sequence",
    4: "Supergiant",
    5: "Hypergiant"
}

# ===============================
# Background Styling
# ===============================
def add_bg_from_url():
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("https://i.ibb.co/qR7D9JX/universe-bg.jpg");
            background-size: cover;
            background-attachment: fixed;
            color: white;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

add_bg_from_url()

# ===============================
# Prediction Function
# ===============================
def predict_star(input_df):
    input_df["Color"] = color_encoder.transform(input_df["Color"])
    input_df["Spectral_Class"] = spectral_encoder.transform(input_df["Spectral_Class"])

    numeric_cols = ["Temperature", "L", "R", "A_M"]
    input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])

    prediction = model.predict(input_df)[0]
    return STAR_TYPES[prediction]

# ===============================
# Sidebar Navigation
# ===============================
st.sidebar.title("🔭 Star Type Classifier")
page = st.sidebar.radio("Choose Option", ["Single Prediction", "Batch Prediction", "Visualization"])

# ===============================
# Single Prediction
# ===============================
if page == "Single Prediction":
    st.title("⭐ Star Type Classification")
    st.markdown("Enter the star parameters below:")

    temp = st.number_input("Temperature (K)", min_value=2000, max_value=40000, value=5778)
    L = st.number_input("Luminosity (L/Lo)", min_value=0.01, max_value=100000, value=1.0)
    R = st.number_input("Radius (R/Ro)", min_value=0.1, max_value=1000.0, value=1.0)
    A_M = st.number_input("Absolute Magnitude", min_value=-10.0, max_value=20.0, value=4.8)

    color = st.selectbox("Color", ["Red", "Blue", "Yellow", "White", "Orange"])
    spectral = st.selectbox("Spectral Class", ["O", "B", "A", "F", "G", "K", "M"])

    if st.button("🔮 Predict Star Type"):
        input_data = pd.DataFrame([[temp, L, R, A_M, color, spectral]],
                                  columns=["Temperature", "L", "R", "A_M", "Color", "Spectral_Class"])
        result = predict_star(input_data)
        st.success(f"🌟 Predicted Star Type: **{result}**")

# ===============================
# Batch Prediction
# ===============================
elif page == "Batch Prediction":
    st.title("📂 Batch Star Type Prediction")
    uploaded_file = st.file_uploader("Upload CSV with columns: Temperature, L, R, A_M, Color, Spectral_Class", type="csv")

    if uploaded_file:
        batch_df = pd.read_csv(uploaded_file)
        st.write("✅ Uploaded Data Preview:", batch_df.head())

        try:
            batch_df["Color"] = color_encoder.transform(batch_df["Color"])
            batch_df["Spectral_Class"] = spectral_encoder.transform(batch_df["Spectral_Class"])
            numeric_cols = ["Temperature", "L", "R", "A_M"]
            batch_df[numeric_cols] = scaler.transform(batch_df[numeric_cols])

            predictions = model.predict(batch_df)
            batch_df["Predicted_Type"] = [STAR_TYPES[p] for p in predictions]

            st.success("🌟 Predictions completed!")
            st.write(batch_df)

            # Download option
            csv = batch_df.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="star_predictions.csv">📥 Download Predictions</a>'
            st.markdown(href, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"❌ Error in prediction: {e}")

# ===============================
# Visualization
# ===============================
elif page == "Visualization":
    st.title("📊 Star Data Visualization")
    st.markdown("Explore relationships between star features.")

    # Example: Temperature vs Luminosity
    try:
        df = pd.read_csv("data/star_preprocessed.csv")  # use your dataset file
        fig, ax = plt.subplots()
        scatter = ax.scatter(df["Temperature"], df["L"], c=df["Type"], cmap="viridis")
        ax.set_xlabel("Temperature (K)")
        ax.set_ylabel("Luminosity (L/Lo)")
        ax.set_title("Temperature vs Luminosity by Star Type")
        st.pyplot(fig)
    except Exception:
        st.warning("⚠️ Dataset not found for visualization. Please upload star_preprocessed.csv to /data.")
