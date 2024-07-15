# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 04:05:27 2024

@author: Rabbiya
"""
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

# Function to load and parse a single .dat file
def load_dat_file(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            label, concentration = parts[0].split(';')
            features = {int(k): float(v) for k, v in (item.split(':') for item in parts[1:])}
            features['label'] = label
            features['concentration'] = concentration
            data.append(features)
    return pd.DataFrame(data)

# Define the path to the directory containing .dat files
data_dir = "data"

# List all .dat files in the directory
dat_files = [os.path.join(data_dir, file) for file in os.listdir(data_dir) if file.endswith('.dat')]

# Load and parse all .dat files
dataframes = [load_dat_file(file) for file in dat_files]

# Concatenate all dataframes
df = pd.concat(dataframes, ignore_index=True)

# Convert 'label' and 'concentration' to numeric
df['label'] = pd.to_numeric(df['label'], errors='coerce')
df['concentration'] = pd.to_numeric(df['concentration'], errors='coerce')

# Drop any rows with NaN values
df = df.dropna()
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 15))
features_to_plot = ['label', 'concentration', 1, 2, 3, 4, 5, 6, 7]  # Choose relevant feature indices

for i, feature in enumerate(features_to_plot):
    sns.histplot(df[feature], bins=30, kde=True, ax=axes[i // 3, i % 3])
    axes[i // 3, i % 3].set_title(f'Distribution of {feature}')

plt.tight_layout()
plt.show()
# Separate features and target
X = df.drop(columns=['label', 'concentration'])
y = df['label']  # or use 'concentration' based on your target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the models
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest Regressor': RandomForestRegressor(n_estimators=100, random_state=42),
    'Support Vector Regressor': SVR(),
    'Gradient Boosting Regressor': GradientBoostingRegressor(n_estimators=100, random_state=42),
    'K-Nearest Neighbors Regressor': KNeighborsRegressor(n_neighbors=5),
    'Decision Tree Regressor': DecisionTreeRegressor(random_state=42),
    'Ridge Regression': Ridge(alpha=1.0),
    'Lasso Regression': Lasso(alpha=0.1)
}

# Fit the models and predict
mse_results = {}
predictions = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mse_results[name] = mse
    predictions[name] = y_pred
    print(f'{name} Mean Squared Error: {mse}')

# Plot Mean Squared Error for all models
plt.figure(figsize=(12, 6))
plt.bar(mse_results.keys(), mse_results.values())
plt.xlabel('Model')
plt.ylabel('Mean Squared Error')
plt.title('Model Comparison - Mean Squared Error')
plt.xticks(rotation=45)
plt.show()

# Scatter plot for actual vs. predicted values for all models
plt.figure(figsize=(15, 15))
for i, (name, y_pred) in enumerate(predictions.items(), start=1):
    plt.subplot(4, 2, i)
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'k--', lw=2)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title(f'Actual vs Predicted - {name}')

plt.tight_layout()
plt.show()

# Residual plots for all models
plt.figure(figsize=(15, 15))
for i, (name, y_pred) in enumerate(predictions.items(), start=1):
    residuals = y_test - y_pred
    plt.subplot(4, 2, i)
    plt.scatter(y_test, residuals, alpha=0.5)
    plt.hlines(y=0, xmin=min(y_test), xmax=max(y_test), color='red', lw=2)
    plt.xlabel('Actual Values')
    plt.ylabel('Residuals')
    plt.title(f'Residuals - {name}')

plt.tight_layout()
plt.show()

