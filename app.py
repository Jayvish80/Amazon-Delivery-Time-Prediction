import streamlit as st
import joblib
import numpy as np

# Load the trained model
# model = joblib.load("best_model.pkl")
model = joblib.load("/content/drive/MyDrive/Amazon Delivery Time Prediction/best_model.pkl")

st.title("🚚 Amazon Delivery Time Prediction App")
st.write("Enter delivery details to predict estimated delivery time.")

# Input fields
distance = st.number_input("Distance (in km)", min_value=0.0, step=0.1)
traffic = st.selectbox("Traffic Condition", ["Low", "Medium", "High"])
weather = st.selectbox("Weather Condition", ["Clear", "Rainy", "Stormy"])
agent_rating = st.slider("Agent Rating", 1.0, 5.0, 4.0)

# Encoding categorical inputs (example mapping)
traffic_map = {"Low": 0, "Medium": 1, "High": 2}
weather_map = {"Clear": 0, "Rainy": 1, "Stormy": 2}

# Prepare features
features = np.array([[distance,
                      traffic_map[traffic],
                      weather_map[weather],
                      agent_rating]])

# Prediction
if st.button("Predict Delivery Time"):
    prediction = model.predict(features)
    st.success(f"Estimated Delivery Time: {prediction[0]:.2f} hours")
