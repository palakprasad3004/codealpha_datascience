import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset
car_data = pd.read_csv("car data.csv")

# Feature engineering
car_data['Current_Year'] = 2025
car_data['Car_Age'] = car_data['Current_Year'] - car_data['Year']
car_data.drop(['Year', 'Current_Year'], axis=1, inplace=True)

# Encoding
label_encoder = LabelEncoder()

car_data['Fuel_Type'] = label_encoder.fit_transform(car_data['Fuel_Type'])
car_data['Selling_type'] = label_encoder.fit_transform(car_data['Selling_type'])
car_data['Transmission'] = label_encoder.fit_transform(car_data['Transmission'])
car_data['Car_Name'] = label_encoder.fit_transform(car_data['Car_Name'])

# Features and target
X = car_data.drop(['Selling_Price'], axis=1)
y = car_data['Selling_Price']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model training
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Evaluation
print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# Visualization
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Prices")
plt.show()