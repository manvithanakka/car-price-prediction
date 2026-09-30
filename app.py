import streamlit as st
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(
    page_title="ML Prediction Hub",
    page_icon="🤖",
    layout="wide"
)

# SIDEBAR
st.sidebar.title("🤖 ML Prediction Hub")

project = st.sidebar.selectbox(
    "Choose a Project",
    [
        "🚗 Car Selling Price Prediction",
        "🏠 Boston House Price Prediction"
    ]
)

# =====================================================
# CAR PRICE PREDICTION
# =====================================================

if project == "🚗 Car Selling Price Prediction":

    st.title("🚗 Car Selling Price Predictor")
    st.write("Machine Learning based used-car price prediction")

    model = joblib.load("car_price_model.pkl")

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

    if st.button("🔮 Predict Selling Price", use_container_width=True):

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

        st.success(
            f"🚗 Estimated Selling Price: ₹{prediction:.2f} Lakhs"
        )

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
        "The model uses vehicle year, showroom price, kilometers driven, "
        "fuel type, seller type, transmission and previous owners."
    )


# =====================================================
# BOSTON HOUSE PRICE PREDICTION
# =====================================================

else:

    st.title("🏠 Boston House Price Prediction")
    st.write(
        "Machine Learning based Boston house price prediction"
    )

    # LOAD BOSTON DATASET
    df = pd.read_csv(
        "Boston_House_Price_Prediction/data/Boston.csv"
    )

    # FEATURES AND TARGET
    X = df.drop("MEDV", axis=1)
    y = df["MEDV"]

    # TRAIN TEST SPLIT
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # MODEL
    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train, y_train)

    # EVALUATION
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, y_pred)

    st.subheader("🏡 Enter House Details")

    inputs = {}

    col1, col2 = st.columns(2)

    for i, column in enumerate(X.columns):

        if i % 2 == 0:

            with col1:

                inputs[column] = st.number_input(
                    column,
                    value=float(X[column].median())
                )

        else:

            with col2:

                inputs[column] = st.number_input(
                    column,
                    value=float(X[column].median())
                )

    st.markdown("---")

    if st.button(
        "🔮 Predict House Price",
        use_container_width=True
    ):

        input_data = pd.DataFrame([inputs])

        prediction = model.predict(input_data)[0]

        st.success(
            f"🏠 Predicted House Price: ${prediction:.2f}k"
        )

    st.markdown("---")

    st.subheader("📊 Model Performance")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("MAE", f"{mae:.2f}")

    with col2:
        st.metric("RMSE", f"{rmse:.2f}")

    with col3:
        st.metric("R² Score", f"{r2:.4f}")

    st.info("Model: Random Forest Regression")