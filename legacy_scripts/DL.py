#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import image
from sklearn.metrics import mean_squared_error

# # ACTUAL CODE

# In[1]:


import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, LSTM
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from ipywidgets import widgets
from IPython.display import display, clear_output
import os

# Define weather categories based on the dataset
weather_categories = ["humid", "sunny", "foggy", "rainy"]  # Adjust if more categories exist
num_classes = len(weather_categories)

# Step 1: Image Classification Model
def build_classification_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(num_classes, activation='softmax')  # Ensure output matches `num_classes`
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Initialize and train the classification model
classification_model = build_classification_model()

# Step 2: Time Series Data Generation (for training the LSTM)
def generate_weather_data(num_days):
    base_temp = random.randint(10, 30)  # base temperature for the city
    temps = base_temp + 5 * np.sin(np.linspace(0, 3 * np.pi, num_days)) + np.random.normal(0, 2, num_days)
    dates = pd.date_range(start="2023-01-01", periods=num_days)
    data = pd.DataFrame({"date": dates, "temperature": temps})
    return data

data = generate_weather_data(365)  # 1 year of daily data
train_data = data['temperature'].values

# Step 3: Prepare data for LSTM model
def prepare_lstm_data(data, look_back=30):
    generator = TimeseriesGenerator(data, data, length=look_back, batch_size=1)
    return generator

# Step 4: Build and train LSTM model for future prediction
def train_lstm_model(train_data):
    look_back = 30
    lstm_train_gen = prepare_lstm_data(train_data, look_back=look_back)
    
    lstm_model = Sequential()
    lstm_model.add(LSTM(50, activation='relu', input_shape=(look_back, 1)))
    lstm_model.add(Dense(1))
    lstm_model.compile(optimizer='adam', loss='mse')
    lstm_model.fit(lstm_train_gen, epochs=5, verbose=1)
    
    return lstm_model

lstm_model = train_lstm_model(train_data)

# Step 5: Predict future weather based on LSTM model
def predict_lstm(model, data, look_back=30, days=30):
    predictions = []
    current_batch = data[-look_back:]
    current_batch = current_batch.reshape((1, look_back, 1))
    
    for _ in range(days):
        pred = model.predict(current_batch)[0]
        predictions.append(pred)
        current_batch = np.append(current_batch[:, 1:, :], [[pred]], axis=1)
    
    return np.array(predictions).flatten()

# Step 6: Image upload and prediction function
def classify_and_predict(image_path):
    # Load and preprocess image
    img = load_img(image_path, target_size=(128, 128))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Classify image
    predictions = classification_model.predict(img_array)
    class_index = np.argmax(predictions)
    weather_type = weather_categories[class_index]
    
    # Display classification result
    print(f"Classified weather condition: {weather_type}")
    
    # Make LSTM predictions based on classified weather
    periods = {"1 month": 30, "6 months": 180, "1 year": 365, "2 years": 730}
    for period_name, days in periods.items():
        lstm_predictions = predict_lstm(lstm_model, train_data, days=days)
        print(f"Predicted temperatures for {period_name}:")
        print(lstm_predictions)
        
        # Plot predictions
        plt.figure(figsize=(14, 7))
        plt.plot(range(len(train_data)), train_data, label="Actual Temperature")
        plt.plot(range(len(train_data), len(train_data) + days), lstm_predictions, label=f"LSTM Prediction ({period_name})", color="orange")
        plt.title(f"Temperature Prediction for {period_name}")
        plt.xlabel("Days")
        plt.ylabel("Temperature (°C)")
        plt.legend()
        plt.show()

# Step 7: Interactive image upload widget
upload_widget = widgets.FileUpload(
    accept='image/*',
    multiple=False
)

def on_image_upload(change):
    clear_output(wait=True)
    display(upload_widget)
    for name, file_info in upload_widget.value.items():
        # Save the uploaded image to disk
        image_path = f"temp_{name}"
        with open(image_path, 'wb') as f:
            f.write(file_info['content'])
        classify_and_predict(image_path)
        os.remove(image_path)  # Remove temp file after classification

upload_widget.observe(on_image_upload, names='value')
display(upload_widget)


# # PYGAL GRAPH READING

# In[5]:

