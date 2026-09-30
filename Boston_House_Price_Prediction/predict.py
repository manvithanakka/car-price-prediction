import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(
    page_title="Boston House Price Prediction",
    page_icon="🏠",
    layout="centered"
)

st.title("🏠 Boston House Price Prediction")
st.write("Enter the house-related features below to predict the house price.")

# Load dataset
df = pd.read_csv("Boston_House_Price_Prediction/data/Boston.csv")

# Separate features and target
X = df.drop("MEDV", axis=1)
y = df["MEDV"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# Model evaluation
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)

st.subheader("Enter House Details")

inputs = {}

for column in X.columns:
    inputs[column] = st.number_input(
        column,
        value=float(X[column].median())
    )

if st.button("🔮 Predict House Price"):

    input_data = pd.DataFrame([inputs])

    prediction = model.predict(input_data)[0]

    st.success(
        f"🏠 Predicted House Price: ${prediction:.2f}k"
    )

st.divider()

st.subheader("📊 Model Performance")

col1, col2, col3 = st.columns(3)

col1.metric("MAE", f"{mae:.2f}")
col2.metric("RMSE", f"{rmse:.2f}")
col3.metric("R² Score", f"{r2:.4f}")

st.caption(
    "Model: Random Forest Regression"
)