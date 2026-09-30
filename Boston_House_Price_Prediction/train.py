# ==========================================
# BOSTON HOUSE PRICE PREDICTION
# ==========================================

# 1. Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


# ==========================================
# 2. LOAD DATASET
# ==========================================

data = pd.read_csv("data/Boston.csv")

print("\nDataset loaded successfully!")
print("Shape of dataset:", data.shape)

print("\nFirst 5 rows:")
print(data.head())


# ==========================================
# 3. BASIC DATA INFORMATION
# ==========================================

print("\nColumn names:")
print(data.columns)

print("\nDataset information:")
print(data.info())

print("\nMissing values:")
print(data.isnull().sum())


# ==========================================
# 4. REMOVE MISSING VALUES
# ==========================================

data = data.dropna()

print("\nShape after removing missing values:")
print(data.shape)


# ==========================================
# 5. SEPARATE FEATURES AND TARGET
# ==========================================

# Usually Boston dataset uses MEDV as target
# MEDV = Median value of owner-occupied homes

if "MEDV" in data.columns:
    X = data.drop("MEDV", axis=1)
    y = data["MEDV"]

elif "medv" in data.columns:
    X = data.drop("medv", axis=1)
    y = data["medv"]

elif "PRICE" in data.columns:
    X = data.drop("PRICE", axis=1)
    y = data["PRICE"]

elif "price" in data.columns:
    X = data.drop("price", axis=1)
    y = data["price"]

else:
    # Assume last column is target
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

print("\nFeatures:")
print(X.columns)

print("\nTarget:")
print(y.name)


# ==========================================
# 6. TRAIN-TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining data:", X_train.shape)
print("Testing data:", X_test.shape)


# ==========================================
# 7. FEATURE SCALING
# ==========================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)


# ==========================================
# 8. LINEAR REGRESSION MODEL
# ==========================================

linear_model = LinearRegression()

linear_model.fit(X_train_scaled, y_train)

linear_predictions = linear_model.predict(X_test_scaled)


# ==========================================
# 9. EVALUATE LINEAR REGRESSION
# ==========================================

linear_mae = mean_absolute_error(
    y_test,
    linear_predictions
)

linear_mse = mean_squared_error(
    y_test,
    linear_predictions
)

linear_rmse = np.sqrt(linear_mse)

linear_r2 = r2_score(
    y_test,
    linear_predictions
)


print("\n====================================")
print("LINEAR REGRESSION RESULTS")
print("====================================")

print("MAE :", linear_mae)
print("MSE :", linear_mse)
print("RMSE:", linear_rmse)
print("R2 Score:", linear_r2)


# ==========================================
# 10. RANDOM FOREST MODEL
# ==========================================

random_forest = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

random_forest.fit(X_train, y_train)

rf_predictions = random_forest.predict(X_test)


# ==========================================
# 11. EVALUATE RANDOM FOREST
# ==========================================

rf_mae = mean_absolute_error(
    y_test,
    rf_predictions
)

rf_mse = mean_squared_error(
    y_test,
    rf_predictions
)

rf_rmse = np.sqrt(rf_mse)

rf_r2 = r2_score(
    y_test,
    rf_predictions
)


print("\n====================================")
print("RANDOM FOREST RESULTS")
print("====================================")

print("MAE :", rf_mae)
print("MSE :", rf_mse)
print("RMSE:", rf_rmse)
print("R2 Score:", rf_r2)


# ==========================================
# 12. MODEL COMPARISON
# ==========================================

print("\n====================================")
print("MODEL COMPARISON")
print("====================================")

print("\nLinear Regression R2:",
      linear_r2)

print("Random Forest R2:",
      rf_r2)


# ==========================================
# 13. SELECT BEST MODEL
# ==========================================

if rf_r2 > linear_r2:
    best_model = random_forest
    best_predictions = rf_predictions

    print("\nBest Model: Random Forest")

else:
    best_model = linear_model
    best_predictions = linear_predictions

    print("\nBest Model: Linear Regression")


# ==========================================
# 14. ACTUAL VS PREDICTED VALUES
# ==========================================

comparison = pd.DataFrame({
    "Actual Price": y_test.values,
    "Predicted Price": best_predictions
})

print("\nActual vs Predicted:")
print(comparison.head(10))


# ==========================================
# 15. VISUALIZATION
# ==========================================

plt.figure(figsize=(8, 6))

plt.scatter(
    y_test,
    best_predictions
)

plt.xlabel("Actual House Price")
plt.ylabel("Predicted House Price")

plt.title("Actual vs Predicted House Prices")

plt.show()