import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Load dataset
df = pd.read_csv("dataset.csv")

print("Dataset loaded successfully!")

# 2. Remove Car_Name
df = df.drop("Car_Name", axis=1)

# 3. Convert categorical columns into numbers
df = pd.get_dummies(
    df,
    columns=["Fuel_Type", "Seller_Type", "Transmission"],
    drop_first=True
)

# 4. Separate input features and target
X = df.drop("Selling_Price", axis=1)
y = df["Selling_Price"]

# 5. Split data into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining data:", X_train.shape)
print("Testing data:", X_test.shape)

# 6. Create Random Forest model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

# 7. Train the model
model.fit(X_train, y_train)

print("\nModel training completed!")

# 8. Make predictions
y_pred = model.predict(X_test)

# 9. Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)

print("\n----- MODEL PERFORMANCE -----")
print("MAE:", mae)
print("RMSE:", rmse)
print("R² Score:", r2)
import joblib

# Save the trained model
joblib.dump(model, "car_price_model.pkl")

print("\nModel saved successfully!")