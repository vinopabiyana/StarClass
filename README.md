⭐ Star Type Classification
This is a Streamlit web app for predicting the type of a star using a trained classification model. Users can input star details manually to predict the star type.

🛠️ Files
app.py : Streamlit application code
star_classifier_model.pkl : Trained star type classification model
scaler.pkl : StandardScaler for numeric features
color_encoder.pkl : LabelEncoder for Color feature
spectral_encoder.pkl : LabelEncoder for Spectral_Class feature
data/star_preprocessed.csv : Dataset after basic preprocessing
data/star_processed.csv : Dataset after full processing (scaling & encoding)
requirements.txt : Python dependencies
.gitignore : Files and folders to ignore in Git
🚀 Deployment on Streamlit Cloud
Fork or clone this repository
Ensure all .pkl and CSV files are included
Go to Streamlit Cloud
Click New App, connect your GitHub repository, and select app.py
Streamlit will automatically install dependencies from requirements.txt
💡 Features
Predict the type of a star based on numeric and categorical features:

Temperature
Luminosity (L)
Radius (R)
Absolute Magnitude (A_M)
Color
Spectral Class
Automatically scales numeric features and encodes categorical features before prediction

Displays the predicted star type

⚙️ Usage
Open the deployed app
Fill in star details in the sidebar form
Click Predict Star Type
View the predicted star type
📄 License
This project is licensed under the MIT License
