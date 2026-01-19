import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os
from model import SolarPredictor


def train_model():
    # 1. Load Data
    data_path = os.path.join("data", "solar_data.csv")
    if not os.path.exists(data_path):
        print("Data not found. Please run data_generator.py first.")
        return

    df = pd.read_csv(data_path)

    # Features and Target
    # We use: hour, month, temperature, irradiance, wind_speed
    feature_cols = ['hour', 'month', 'temperature', 'irradiance', 'wind_speed']
    target_col = 'power_output'

    X = df[feature_cols].values
    y = df[target_col].values.reshape(-1, 1)

    # 2. Preprocessing
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)

    # Save scalers for inference
    os.makedirs("models", exist_ok=True)
    joblib.dump(scaler_X, "models/scaler_X.pkl")
    joblib.dump(scaler_y, "models/scaler_y.pkl")

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_scaled, test_size=0.2, random_state=42)

    # Convert to Tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test)

    # 3. Model Setup
    input_dim = len(feature_cols)
    model = SolarPredictor(input_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 4. Training Loop
    epochs = 100
    batch_size = 32

    print("Starting training...")
    for epoch in range(epochs):
        model.train()

        # Mini-batch training (simplified manual batching)
        permutation = torch.randperm(X_train_tensor.size()[0])

        for i in range(0, X_train_tensor.size()[0], batch_size):
            indices = permutation[i:i+batch_size]
            batch_x, batch_y = X_train_tensor[indices], y_train_tensor[indices]

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

        if (epoch+1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                test_outputs = model(X_test_tensor)
                test_loss = criterion(test_outputs, y_test_tensor)
            print(
                f'Epoch [{epoch+1}/{epochs}], Train Loss: {loss.item():.4f}, Test Loss: {test_loss.item():.4f}')

    # 5. Save Model
    torch.save(model.state_dict(), "models/solar_model.pth")
    print("Model saved to models/solar_model.pth")


if __name__ == "__main__":
    train_model()
