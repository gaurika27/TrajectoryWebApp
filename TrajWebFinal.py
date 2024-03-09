#!/usr/bin/env python
# coding: utf-8

# In[16]:


#!pip install tensorflow


# In[17]:


# !pip install --upgrade tensorflow


# In[18]:


import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from tensorflow import keras
from tensorflow.keras import layers


# In[19]:


# Load your satellite trajectory dataset (replace 'your_dataset.csv' with the actual file name)
# Make sure your dataset includes columns like 'initial_velocity', 'launch_angle', etc.
df = pd.read_csv('satellite_trajectory_dataset.csv')


# In[20]:


# Define features (X) and target variable (y)
features = df[['initial_velocity', 'launch_angle', 'atmospheric_conditions']]
target = df['trajectory_optimization']  


# In[21]:


# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)


# In[22]:


# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[23]:


# Build the neural network model
model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    layers.Dense(32, activation='relu'),
    layers.Dense(1)  # Output layer
])


# In[24]:


# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')


# In[25]:


# Train the model
model.fit(X_train_scaled, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=2)


# In[26]:


# Evaluate the model on the test set
predictions = model.predict(X_test_scaled)
mse = mean_squared_error(y_test, predictions)
print(f'Mean Squared Error: {mse}')


pickle.dump(model, open('model.pkl','wb'))
model=pickle.load(open('model.pkl','rb'))
print(model.predict([[200,400]]))

# In[27]:


# Visualize results (for simplicity, you can modify this based on your actual data)
import matplotlib.pyplot as plt
plt.scatter(X_test['initial_velocity'], y_test, color='black', label='Actual')
plt.scatter(X_test['initial_velocity'], predictions, color='blue', label='Predicted')
plt.xlabel('Initial Velocity')
plt.ylabel('Optimized Trajectory')
plt.legend()
plt.show()

