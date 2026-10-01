# 🤖 ML Prediction Hub

🚀 **Live Demo :** https://manvitha-car-price-predictor.streamlit.app/

## 📌 About the Project

This repository contains a collection of Machine Learning projects focused on solving real-world prediction problems using data preprocessing, regression techniques, model training, and evaluation.

The projects are integrated into a single **Streamlit-based ML Prediction Hub**, providing an interactive platform where users can select a prediction task, enter the required inputs, and obtain predictions in real time.

### 🚗 Car Selling Price Prediction

The Car Selling Price Prediction project estimates the resale value of a used car based on important factors such as **manufacturing year, showroom price, kilometers driven, fuel type, seller type, transmission, and previous owners**.

A **Random Forest Regression** model is trained on historical vehicle data after preprocessing and transforming categorical features. The model is evaluated using **Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and R² Score**.

### 🏠 Boston House Price Prediction

The Boston House Price Prediction project estimates residential property prices using various housing-related characteristics such as **crime rate, number of rooms, accessibility, taxation-related factors, and other relevant features**.

The project follows a complete Machine Learning workflow including **data preprocessing, train-test splitting, model training, prediction, and performance evaluation** using a Random Forest Regression model.

### 🔑 Key Features

- 🚗 Used-car selling price prediction
- 🏠 Boston house price prediction
- 🤖 Random Forest Regression
- 📊 Data preprocessing and feature transformation
- 📈 Model performance evaluation
- 🔮 Real-time predictions
- 🖥️ Interactive Streamlit web application
- 🔗 Both projects accessible through a single live application

### 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Streamlit

## 📂 Project Structure

```text
car-price-prediction/
│
├── app.py
├── train_model.py
├── dataset.csv
├── car_price_model.pkl
├── requirements.txt
├── README.md
│
└── Boston_House_Price_Prediction/
    ├── data/
    ├── train.py
    ├── predict.py
    └── requirements.txt
```

## 📊 Machine Learning Workflow

**Data Collection → Data Preprocessing → Feature Transformation → Train-Test Split → Model Training → Prediction → Performance Evaluation → Streamlit Deployment**

## 🌐 Deployment

The projects are deployed together using **Streamlit**, allowing users to access both prediction applications through a single web interface.

🚀 **Live Demo:** [ML Prediction Hub](https://manvitha-car-price-predictor.streamlit.app/)
