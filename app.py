```python
import streamlit as st
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="ML Prediction Hub",
    page_icon="🤖",
    layout="wide"
)

# ---------------- SIDEBAR ----------------
st.sidebar.title("🤖 ML Prediction Hub")

project = st.sidebar.radio(
    "Select Project",
    [
        "🚗 Car Selling Price Prediction",
        "🏠 Boston House Price Prediction"
    ]
)

# ============================================================
#                    CAR PRICE PREDICTION
# ============================================================

if project == "🚗 Car Selling Price Prediction":

    # ---------------- LOAD CAR MODEL ----------------
    model = joblib.load("car_price_model.pkl")

    # ---------------- CUSTOM CSS ----------------
    st.markdown("""
    <style>

    .main {
        background-color: #f5f7fb;
    }

    .title {
        text-align: center;
        font-size: 42px;
        font-weight: bold;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
        color: #666666;
        margin-bottom: 30px;
    }

    .result {
        background: linear-gradient(135deg, #667eea, #764ba2);
        padding: 30px;
        border-radius: 18px;
        text-align: center;
        color: white;
        margin-top: 25px;
    }

    .result-title {
        font-size: 20px;
    }

    .result-price {
        font-size: 40px;
        font-weight: bold;
    }

    .footer {
        text-align: center;
        color: #777777;
        margin-top: 40px;
    }

    </style>
    """, unsafe_allow_html=True)

    # ---------------- HEADER ----------------
    st.markdown(
        '<div class="title">🚗 Car Selling Price Predictor</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">Machine Learning based used-car price prediction</div>',
        unsafe_allow_html=True
    )

    st.markdown("---")

    # ---------------- INPUT SECTION ----------------
    st.subheader("🚘 Enter Car Details")

    col1, col2 = st.columns(2)

    with col1:

        year = st.number_input(
            "📅 Manufacturing Year",
            min_value=1990,
            max_value=2026,
            value=2018,
            step=1
        )

        present_price = st.number_input(
            "💰 Current / Showroom Price (Lakhs)",
            min_value=0.0,
            value=5.0,
            step=0.1
        )

        kms_driven = st.number_input(
            "🛣️ Kilometers Driven",
            min_value=0,
            value=30000,
            step=1000
        )

        owner = st.number_input(
            "👤 Previous Owners",
            min_value=0,
            max_value=3,
            value=0,
            step=1
        )

    with col2:

        fuel_type = st.selectbox(
            "⛽ Fuel Type",
            ["Petrol", "Diesel", "CNG"]
        )

        seller_type = st.selectbox(
            "🏪 Seller Type",
            ["Dealer", "Individual"]
        )

        transmission = st.selectbox(
            "⚙️ Transmission",
            ["Manual", "Automatic"]
        )

    st.markdown("---")

    # ---------------- PREDICTION ----------------
    center = st.columns([1, 2, 1])

    with center[1]:

        predict = st.button(
            "🔮 Predict Selling Price",
            use_container_width=True
        )

    if predict:

        input_data = pd.DataFrame({
            "Year": [year],
            "Present_Price": [present_price],
            "Kms_Driven": [kms_driven],
            "Owner": [owner],
            "Fuel_Type_Diesel": [fuel_type == "Diesel"],
            "Fuel_Type_Petrol": [fuel_type == "Petrol"],
            "Seller_Type_Individual": [seller_type == "Individual"],
            "Transmission_Manual": [transmission == "Manual"]
        })

        prediction = model.predict(input_data)[0]

        st.markdown(
            f"""
            <div class="result">
                <