import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from ipywidgets import widgets
from IPython.display import display, clear_output

# 20 famous cities
cities = [
    "New York", "Paris", "Tokyo", "Dubai", "London", "Sydney", "San Francisco", 
    "Rome", "Mumbai", "Beijing", "Moscow", "Los Angeles", "Toronto", 
    "Cape Town", "Istanbul", "Berlin", "Rio de Janeiro", "Singapore", 
    "Seoul", "Bangkok"
]

# Generate random weather data for temperature, humidity, rainfall, and wind speed
def generate_weather_data(num_days):
    base_temp = random.randint(10, 30)
    temps = base_temp + 5 * np.sin(np.linspace(0, 3 * np.pi, num_days)) + np.random.normal(0, 2, num_days)
    humidity = 50 + 10 * np.sin(np.linspace(0, 3 * np.pi, num_days)) + np.random.normal(0, 5, num_days)
    rainfall = np.clip(3 + 2 * np.sin(np.linspace(0, 3 * np.pi, num_days)) + np.random.normal(0, 1, num_days), 0, None)
    wind_speed = np.clip(10 + 3 * np.sin(np.linspace(0, 3 * np.pi, num_days)) + np.random.normal(0, 1.5, num_days), 0, None)
    dates = pd.date_range(start="2023-01-01", periods=num_days)
    data = pd.DataFrame({"date": dates, "temperature": temps, "humidity": humidity, "rainfall": rainfall, "wind_speed": wind_speed})
    return data

# Generate data
data = generate_weather_data(365)  # 1 year of daily data
train_data = data[["temperature", "humidity", "rainfall", "wind_speed"]].values

# Prepare data for LSTM model
def prepare_lstm_data(data, look_back=30):
    generator = TimeseriesGenerator(data, data, length=look_back, batch_size=1)
    return generator

# Build and train LSTM model
def train_lstm_model(train_data):
    look_back = 30
    lstm_train_gen = prepare_lstm_data(train_data, look_back=look_back)
    
    lstm_model = Sequential()
    lstm_model.add(LSTM(50, activation='relu', input_shape=(look_back, train_data.shape[1])))
    lstm_model.add(Dense(train_data.shape[1]))
    lstm_model.compile(optimizer='adam', loss='mse')
    lstm_model.fit(lstm_train_gen, epochs=5, verbose=1)
    
    return lstm_model

# Train the LSTM model
lstm_model = train_lstm_model(train_data)

# Predict using LSTM for a specified number of days
def predict_lstm(model, data, look_back=30, days=30):
    predictions = []
    current_batch = data[-look_back:]
    current_batch = current_batch.reshape((1, look_back, data.shape[1]))
    
    for _ in range(days):
        pred = model.predict(current_batch)[0]
        predictions.append(pred)
        current_batch = np.append(current_batch[:, 1:, :], [[pred]], axis=1)
    
    return np.array(predictions)  # Return 2D array (days x features)

# Train Linear Regression models for each feature
lr_models = {}
for idx, feature in enumerate(["temperature", "humidity", "rainfall", "wind_speed"]):
    lr_model = LinearRegression()
    lr_model.fit(np.arange(len(train_data)).reshape(-1, 1), train_data[:, idx])
    lr_models[feature] = lr_model

# Predict using Linear Regression for each feature
def predict_linear_regression(models, days=30):
    future_days = np.arange(len(train_data), len(train_data) + days).reshape(-1, 1)
    predictions = {
        feature: model.predict(future_days)
        for feature, model in models.items()
    }
    return predictions  # Return dictionary of predictions

# Plot predictions for each weather feature
def plot_predictions(city, lstm_preds, lr_preds, actual_data, period):
    time_range = range(len(actual_data), len(actual_data) + period)
    feature_keys = ["temperature", "humidity", "rainfall", "wind_speed"]

    for idx, feature in enumerate(["Temperature", "Humidity", "Rainfall", "Wind Speed"]):
        plt.figure(figsize=(14, 7))
        plt.plot(actual_data[-60:, idx], label=f"Actual {feature}")
        plt.plot(time_range, lstm_preds[:, idx], color="orange", label=f"LSTM Prediction ({period} days)")
        plt.plot(time_range, lr_preds[feature_keys[idx]], color="green", label=f"Linear Regression Prediction ({period} days)")
        plt.title(f"{feature} Forecast for {city}")
        plt.xlabel("Days")
        plt.ylabel(feature)
        plt.legend()
        plt.show()

# Provide explanations for each prediction plot
def explain_graphs(city, period):
    explanations = {
        "Temperature": f"The forecast shows the expected temperature trends for {city} over the next {period} days.",
        "Humidity": f"Humidity predictions indicate the moisture levels in the air for {city}, helping gauge comfort levels.",
        "Rainfall": f"The rainfall forecast provides an outlook on potential precipitation, useful for outdoor planning in {city}.",
        "Wind Speed": f"Wind speed predictions suggest how windy conditions might be in {city}, valuable for activities and safety."
    }
    for feature, explanation in explanations.items():
        print(f"\n{feature} Prediction Explanation:\n{explanation}")

# Define function to handle city and period selection and display predictions
def on_get_prediction_clicked(_):
    city = city_widget.value
    period = time_widget.value
    if city in cities:
        print(f"\nPredicted Weather for {city}:\n")
        days = {"1 month": 30, "6 months": 180, "1 year": 365, "2 years": 730}[period]

        # Generate LSTM predictions
        lstm_predictions = predict_lstm(lstm_model, train_data, days=days)
        
        # Generate Linear Regression predictions
        lr_predictions = predict_linear_regression(lr_models, days=days)
        
        # Plot predictions
        plot_predictions(city, lstm_predictions, lr_predictions, train_data, period=days)
        
        # Provide explanations
        explain_graphs(city, period)
        
        # Display final predicted values
        print("\nFinal Predicted Values (LSTM & Linear Regression):")
        print(f"Temperature: {lstm_predictions[-1, 0]:.2f}°C (LSTM), {lr_predictions['temperature'][-1]:.2f}°C (Linear Regression)")
        print(f"Humidity: {lstm_predictions[-1, 1]:.2f}% (LSTM), {lr_predictions['humidity'][-1]:.2f}% (Linear Regression)")
        print(f"Rainfall: {lstm_predictions[-1, 2]:.2f}mm (LSTM), {lr_predictions['rainfall'][-1]:.2f}mm (Linear Regression)")
        print(f"Wind Speed: {lstm_predictions[-1, 3]:.2f} km/h (LSTM), {lr_predictions['wind_speed'][-1]:.2f} km/h (Linear Regression)")
    else:
        print("Please select a valid city and time period.")

# Interactive widgets for city, period selection, and prediction button
city_widget = widgets.Dropdown(options=cities, description='Select City:')
time_widget = widgets.Dropdown(options=["1 month", "6 months", "1 year", "2 years"], description='Select Period:')
prediction_button = widgets.Button(description="Get Prediction")

# Display widgets and update based on button click
prediction_button.on_click(on_get_prediction_clicked)

# Display widgets initially
display(city_widget, time_widget, prediction_button)
