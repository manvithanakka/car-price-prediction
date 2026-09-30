import streamlit as st
import pandas as pd
import joblib

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Car Price Predictor",
    page_icon="🚗",
    layout="wide"
)

# ---------------- LOAD MODEL ----------------
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

.card {
    background-color: white;
    padding: 25px;
    border-radius: 15px;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.08);
    margin-bottom: 20px;
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
            <div class="result-title">Estimated Selling Price</div>
            <div class="result-price">₹{prediction:.2f} Lakhs</div>
        </div>
        """,
        unsafe_allow_html=True
    )

# ---------------- MODEL INFORMATION ----------------

st.markdown("---")

st.subheader("📊 Model Information")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Algorithm", "Random Forest")

with col2:
    st.metric("Training Records", "240")

with col3:
    st.metric("Test Records", "61")

st.info(
    "The model was trained using vehicle year, showroom price, "
    "kilometers driven, fuel type, seller type, transmission, "
    "and previous owners."
)

# ---------------- FOOTER ----------------

st.markdown(
    '<div class="footer">Built using Python, Pandas, Scikit-learn and Streamlit 🚗</div>',
    unsafe_allow_html=True
)