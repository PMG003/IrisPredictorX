import streamlit as st
import pandas as pd
import numpy as np
import pickle

# App Title and Description
st.set_page_config(page_title="🌸 Iris Species Classifier", layout="centered")
st.title("🌸 Iris Species Classifier")
st.markdown("""
Predict the species of an Iris flower using a trained Machine Learning model (Random Forest Classifier).
Provide the flower's characteristics below and get an instant prediction!
""")

# Load the trained model
with open('xgboost_regressor_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Define feature input sliders
st.subheader("Enter Flower Details")

col1, col2 = st.columns(2)

with col1:
    sepal_length = st.slider("Sepal Length (cm)", min_value=4.0, max_value=8.0, value=5.8, step=0.1)
    sepal_width = st.slider("Sepal Width (cm)", min_value=2.0, max_value=4.5, value=3.0, step=0.1)

with col2:
    petal_width = st.slider("Petal Width (cm)", min_value=0.1, max_value=2.5, value=1.0, step=0.1)
    species_encoded = st.selectbox("Species Encoded", [0, 1, 2], help="Use values as per label encoding")

# Prediction
if st.button("🌼 Predict Species"):
    input_features = pd.DataFrame([[
        sepal_length, sepal_width, petal_width
    ]], columns=['SepalLengthCm', 'SepalWidthCm', 'PetalWidthCm'])

    prediction = model.predict(input_features)[0]

    # Reverse mapping of label if needed
    species_map = {0: "Iris-setosa", 1: "Iris-versicolor", 2: "Iris-virginica"}
    predicted_species = species_map.get(prediction, "Unknown")

    st.success(f"🌟 Predicted Species: **{predicted_species}**")
    st.caption("Note: Model trained using Random Forest Classifier on the Iris dataset.")

# Footer
st.markdown("---")
st.markdown("Built with ❤️ by Basant | Model: `IrisPredictorX`")
