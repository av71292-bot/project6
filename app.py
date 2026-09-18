import streamlit as st
import numpy as np
import pandas as pd
import joblib
import warnings

warnings.filterwarnings("ignore")  

# Page Config
st.set_page_config(page_title="Rossmann Store Sales & Customer Prediction", layout="centered")

st.title("📊 Rossmann Store Sales & Customer Dashboard")
st.write("Enter store details below to predict daily sales and estimated customer footfall.")

# Lazy load TensorFlow
@st.cache_resource
def load_assets():
    import tensorflow as tf
    model = tf.keras.models.load_model("lstm_model.keras")
    scaler_X = joblib.load("scaler_X.pkl")
    scaler_y = joblib.load("scaler_y.pkl")
    return model, scaler_X, scaler_y

with st.spinner("Loading AI Model and Scalers... Please wait"):
    try:
        model, scaler_X, scaler_y = load_assets()
        st.success("Model and Scalers Loaded Successfully!")
    except Exception as e:
        st.error(f"Error loading model/scalers: {e}")

# User Inputs Section (Friendly UI)
st.header("Store Parameters")

store_id = st.number_input("Store ID", min_value=1, max_value=1115, value=1)

promo_choice = st.selectbox("Is Promo Active?", ["No", "Yes"])
school_holiday_choice = st.selectbox("School Holiday?", ["No", "Yes"])
state_holiday_choice = st.selectbox("State Holiday?", ["No", "Yes"])

day_of_week = st.slider("Day of Week (1=Mon, 7=Sun)", 1, 7, 1)

# Map Yes/No to 1/0
promo = 1 if promo_choice == "Yes" else 0
school_holiday = 1 if school_holiday_choice == "Yes" else 0
state_holiday = 1 if state_holiday_choice == "Yes" else 0

# Prediction Logic
if st.button("Predict Sales & Customers"):
    try:
        # 1. Scaler X ke liye 25 features ka array banayein (1 row, 25 columns)
        num_features = scaler_X.scale_.shape[0] # Auto-detects 25 features
        input_data = np.zeros((1, num_features))
        
        # User inport values set karein (pehli kuch columns par)
        input_data[0, 0] = store_id
        input_data[0, 1] = day_of_week
        input_data[0, 2] = promo
        input_data[0, 3] = state_holiday
        input_data[0, 4] = school_holiday
        
        # 2. Scale the input (25 features)
        scaled_input_2d = scaler_X.transform(input_data)
        
        # 3. Reshape for LSTM (1 sample, 1 timesteps, 25 features)
        scaled_input_3d = scaled_input_2d.reshape(1, 1, num_features)
        
        # 4. Predict using Model
        prediction_scaled = model.predict(scaled_input_3d)
        predicted_sales = float(scaler_y.inverse_transform(prediction_scaled)[0][0])
        
        # Estimate Customers based on average spend (~$9.5 per customer)
        estimated_customers = int(predicted_sales / 9.5) if predicted_sales > 0 else 0
        
        st.markdown("---")
        st.header("📈 Prediction Results")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="Predicted Sales Amount", value=f"${predicted_sales:,.2f}")
        with col2:
            st.metric(label="Predicted Customer Count", value=f"{estimated_customers:,}")

        # Plot as required in Task 3
        st.subheader("Visualization")
        chart_data = pd.DataFrame({
            "Metric": ["Predicted Sales ($)", "Estimated Customers"],
            "Value": [predicted_sales, estimated_customers]
        })
        st.bar_chart(chart_data.set_index("Metric"))

        # Download CSV Option as required in Task 3
        df_download = pd.DataFrame([{
            "Store_ID": store_id,
            "DayOfWeek": day_of_week,
            "Promo": promo,
            "SchoolHoliday": school_holiday,
            "StateHoliday": state_holiday,
            "Predicted_Sales": round(predicted_sales, 2),
            "Predicted_Customers": estimated_customers
        }])
        
        csv = df_download.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Predictions CSV",
            data=csv,
            file_name="rossmann_predictions.csv",
            mime="text/csv"
        )
        
    except Exception as e:
        st.error(f"Prediction Error: {e}")