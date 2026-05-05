import streamlit as st
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image

# ----------------- Configuration & Caching -----------------
st.set_page_config(page_title="Weather Forecast & Classification AI", layout="wide")

@st.cache_data
def generate_weather_data(num_days, features='all'):
    base_temp = random.randint(10, 30)
    temps = base_temp + 5 * np.sin(np.linspace(0, 3 * np.pi, num_days)) + np.random.normal(0, 2, num_days)
    dates = pd.date_range(start="2023-01-01", periods=num_days)
    
    if features == 'all':
        humidity = 50 + 10 * np.sin(np.linspace(0, 3 * np.pi, num_days)) + np.random.normal(0, 5, num_days)
        rainfall = np.clip(3 + 2 * np.sin(np.linspace(0, 3 * np.pi, num_days)) + np.random.normal(0, 1, num_days), 0, None)
        wind_speed = np.clip(10 + 3 * np.sin(np.linspace(0, 3 * np.pi, num_days)) + np.random.normal(0, 1.5, num_days), 0, None)
        return pd.DataFrame({"date": dates, "temperature": temps, "humidity": humidity, "rainfall": rainfall, "wind_speed": wind_speed})
    else:
        return pd.DataFrame({"date": dates, "temperature": temps})

@st.cache_resource
def train_lstm_multivariate(train_data):
    look_back = 30
    generator = TimeseriesGenerator(train_data, train_data, length=look_back, batch_size=1)
    
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(look_back, train_data.shape[1])))
    model.add(Dense(train_data.shape[1]))
    model.compile(optimizer='adam', loss='mse')
    
    with st.spinner("Training LSTM Model for Forecasting..."):
        model.fit(generator, epochs=5, verbose=0)
    return model

@st.cache_resource
def train_lr_models(train_data):
    lr_models = {}
    features = ["temperature", "humidity", "rainfall", "wind_speed"]
    for idx, feature in enumerate(features):
        model = LinearRegression()
        model.fit(np.arange(len(train_data)).reshape(-1, 1), train_data[:, idx])
        lr_models[feature] = model
    return lr_models

@st.cache_resource
def build_classification_model():
    num_classes = 4 # humid, sunny, foggy, rainy
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    # Note: Model is initialized but not trained on a real dataset here, 
    # it behaves as a placeholder matching the original DL.py structure.
    return model

@st.cache_resource
def train_lstm_univariate(train_data):
    look_back = 30
    generator = TimeseriesGenerator(train_data, train_data, length=look_back, batch_size=1)
    
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(look_back, 1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    
    with st.spinner("Training LSTM Model for Image Classification context..."):
        model.fit(generator, epochs=5, verbose=0)
    return model

# ----------------- Helper Functions -----------------
def predict_lstm_multi(model, data, look_back=30, days=30):
    predictions = []
    current_batch = data[-look_back:]
    current_batch = current_batch.reshape((1, look_back, data.shape[1]))
    
    for _ in range(days):
        pred = model.predict(current_batch, verbose=0)[0]
        predictions.append(pred)
        current_batch = np.append(current_batch[:, 1:, :], [[pred]], axis=1)
    return np.array(predictions)

def predict_lr(models, train_data_len, days=30):
    future_days = np.arange(train_data_len, train_data_len + days).reshape(-1, 1)
    predictions = {feature: model.predict(future_days) for feature, model in models.items()}
    return predictions

def predict_lstm_uni(model, data, look_back=30, days=30):
    predictions = []
    current_batch = data[-look_back:]
    current_batch = current_batch.reshape((1, look_back, 1))
    
    for _ in range(days):
        pred = model.predict(current_batch, verbose=0)[0]
        predictions.append(pred)
        current_batch = np.append(current_batch[:, 1:, :], [[pred]], axis=1)
    return np.array(predictions).flatten()

# ----------------- UI Layout -----------------

st.sidebar.title("Navigation")
app_mode = st.sidebar.radio("Select Mode", ["Weather Forecasting", "Weather Image Classification"])

if app_mode == "Weather Forecasting":
    st.title("🌍 Weather Forecasting AI")
    st.write("Predict future weather trends using LSTM and Linear Regression models.")
    
    cities = ["New York", "Paris", "Tokyo", "Dubai", "London", "Sydney", "San Francisco", 
              "Rome", "Mumbai", "Beijing", "Moscow", "Los Angeles", "Toronto", 
              "Cape Town", "Istanbul", "Berlin", "Rio de Janeiro", "Singapore", 
              "Seoul", "Bangkok"]
    
    col1, col2 = st.columns(2)
    with col1:
        city = st.selectbox("Select City", cities)
    with col2:
        period_option = st.selectbox("Select Period", ["1 month", "6 months", "1 year", "2 years"])
        period_days = {"1 month": 30, "6 months": 180, "1 year": 365, "2 years": 730}[period_option]

    if st.button("Get Prediction"):
        # Load Data & Models
        data = generate_weather_data(365, 'all')
        train_data = data[["temperature", "humidity", "rainfall", "wind_speed"]].values
        
        lstm_model = train_lstm_multivariate(train_data)
        lr_models = train_lr_models(train_data)
        
        # Predictions
        lstm_preds = predict_lstm_multi(lstm_model, train_data, days=period_days)
        lr_preds = predict_lr(lr_models, len(train_data), days=period_days)
        
        st.subheader(f"Forecast for {city} ({period_option})")
        
        # Plotting
        time_range = range(len(train_data), len(train_data) + period_days)
        feature_names = ["Temperature", "Humidity", "Rainfall", "Wind Speed"]
        feature_keys = ["temperature", "humidity", "rainfall", "wind_speed"]
        
        for idx, feature in enumerate(feature_names):
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(train_data[-60:, idx], label=f"Actual {feature}")
            ax.plot(time_range, lstm_preds[:, idx], color="orange", label=f"LSTM Prediction")
            ax.plot(time_range, lr_preds[feature_keys[idx]], color="green", label=f"LR Prediction")
            ax.set_title(f"{feature} Forecast")
            ax.set_xlabel("Days")
            ax.set_ylabel(feature)
            ax.legend()
            st.pyplot(fig)
            plt.close(fig)
            
        st.subheader("Final Predicted Values (Last Day of Forecast)")
        col_a, col_b, col_c, col_d = st.columns(4)
        col_a.metric("Temperature", f"{lstm_preds[-1, 0]:.1f}°C")
        col_b.metric("Humidity", f"{lstm_preds[-1, 1]:.1f}%")
        col_c.metric("Rainfall", f"{lstm_preds[-1, 2]:.1f}mm")
        col_d.metric("Wind Speed", f"{lstm_preds[-1, 3]:.1f} km/h")

elif app_mode == "Weather Image Classification":
    st.title("☁️ Weather Image Classification AI")
    st.write("Upload an image of the sky/weather, classify it, and predict future temperatures.")
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        if st.button("Classify & Predict"):
            weather_categories = ["humid", "sunny", "foggy", "rainy"]
            
            # Prepare image for model
            img = image.resize((128, 128))
            img_array = img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            # Classify
            class_model = build_classification_model()
            predictions = class_model.predict(img_array, verbose=0)
            class_index = np.argmax(predictions)
            weather_type = weather_categories[class_index]
            
            st.success(f"Classified weather condition: **{weather_type.upper()}**")
            
            # Temperature Forecasting
            st.subheader("Temperature Trend Analysis")
            data = generate_weather_data(365, 'temp_only')
            train_data = data['temperature'].values
            
            lstm_uni_model = train_lstm_univariate(train_data)
            
            period_option = st.selectbox("Select Forecast Period", ["1 month", "6 months", "1 year", "2 years"])
            period_days = {"1 month": 30, "6 months": 180, "1 year": 365, "2 years": 730}[period_option]
            
            lstm_preds = predict_lstm_uni(lstm_uni_model, train_data, days=period_days)
            
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(range(len(train_data)), train_data, label="Actual Temperature")
            ax.plot(range(len(train_data), len(train_data) + period_days), lstm_preds, label=f"LSTM Prediction", color="orange")
            ax.set_title(f"Temperature Prediction for {period_option}")
            ax.set_xlabel("Days")
            ax.set_ylabel("Temperature (°C)")
            ax.legend()
            st.pyplot(fig)
            plt.close(fig)
