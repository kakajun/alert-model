import torch
import joblib
import numpy as np
import os
from model import SolarPredictor

def predict(hour, month, temperature, irradiance, wind_speed):
    # Load resources
    if not os.path.exists("models/solar_model.pth"):
        print("Model not found. Please run train.py first.")
        return

    scaler_X = joblib.load("models/scaler_X.pkl")
    scaler_y = joblib.load("models/scaler_y.pkl")
    
    # Prepare input
    input_data = np.array([[hour, month, temperature, irradiance, wind_speed]])
    input_scaled = scaler_X.transform(input_data)
    input_tensor = torch.FloatTensor(input_scaled)
    
    # Load Model
    # We need to know input_dim, which is 5 based on our training script
    model = SolarPredictor(input_dim=5)
    model.load_state_dict(torch.load("models/solar_model.pth"))
    model.eval()
    
    # Predict
    with torch.no_grad():
        prediction_scaled = model(input_tensor)
        prediction = scaler_y.inverse_transform(prediction_scaled.numpy())
        
    return max(0.0, prediction[0][0])

if __name__ == "__main__":
    print("Solar Power Prediction Demo")
    print("---------------------------")
    
    # Example inputs
    inputs = [
        # Noon, Summer, Hot, High Sun, Windy
        (12, 6, 30.0, 900.0, 5.0),
        # Night
        (0, 1, 5.0, 0.0, 2.0),
        # Morning, Winter
        (9, 12, 10.0, 300.0, 3.0)
    ]
    
    for h, m, t, i, w in inputs:
        p = predict(h, m, t, i, w)
        print(f"Input: Hour={h}, Month={m}, Temp={t}C, Irr={i}W/m2, Wind={w}m/s")
        print(f"Predicted Power: {p:.2f} kW")
        print("-" * 30)
