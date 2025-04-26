import streamlit as st
import pandas as pd
import numpy as np
import pickle

# App Configuration
st.set_page_config(page_title="🌸 Iris Species Classifier", layout="centered")
st.title("🌸 Iris Species Classifier")
st.markdown("""
Predict the species of an Iris flower using a trained Machine Learning model (Random Forest Classifier).
Fill in the flower's measurements and click the button to see the prediction.
""")

# Load the trained model
@st.cache_resource
def load_model():
    with open('iris_rf_model.pkl', 'rb') as model_file:
        return pickle.load(model_file)

model = load_model()

# Input Features
st.subheader("🌿 Input Flower Features")
col1, col2 = st.columns(2)

with col1:
    sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.8, 0.1)
    sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.0, 0.1)

with col2:
    petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 4.0, 0.1)
    petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 1.0, 0.1)

# Predict Button
if st.button("🌼 Predict Species"):
    input_data = pd.DataFrame([[
        sepal_length, sepal_width, petal_length, petal_width
    ]], columns=['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)'])

    try:
        prediction = model.predict(input_data)[0]
        species_map = {0: "Iris-setosa", 1: "Iris-versicolor", 2: "Iris-virginica"}
        predicted_species = species_map.get(prediction, "Unknown")

        st.success(f"🌟 Predicted Species: **{predicted_species}**")
    except Exception as e:
        st.error(f"⚠️ Error during prediction: {str(e)}")

# Footer
st.markdown("---")
st.markdown("Built with ❤️ by Basant | Model: `IrisPredictorX`")
