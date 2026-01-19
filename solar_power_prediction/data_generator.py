import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

def generate_solar_data(start_date, days=365, interval_minutes=60):
    """
    Generates synthetic solar power data based on physical principles.
    """
    date_range = [start_date + timedelta(minutes=i*interval_minutes)
                  for i in range(days * 24 * (60 // interval_minutes))]

    n_samples = len(date_range)

    df = pd.DataFrame({'timestamp': date_range})
    df['hour'] = df['timestamp'].dt.hour
    df['month'] = df['timestamp'].dt.month
    df['day_of_year'] = df['timestamp'].dt.dayofyear

    # Simulate Irradiance (Bell curve during the day, seasonally adjusted)
    # Peak at noon (hour 12), zero at night
    # Seasonality: Higher in summer (approx month 6 in northern hemisphere), lower in winter

    def get_irradiance(row):
        hour = row['hour']
        day = row['day_of_year']

        # Seasonal factor (simple sinusoid peaking in mid-year)
        season_factor = 1 + 0.4 * np.sin(2 * np.pi * (day - 80) / 365)

        # Daily pattern (Gaussian-like)
        if 6 <= hour <= 18:
            # Normalized hour from -1 to 1 during daylight
            h_norm = (hour - 12) / 6
            daily_factor = np.exp(-2 * h_norm**2)
            # Add random cloud cover noise
            noise = np.random.uniform(0.8, 1.0) if np.random.random() > 0.2 else np.random.uniform(0.1, 0.6)
            return 1000 * daily_factor * season_factor * noise
        else:
            return 0.0

    df['irradiance'] = df.apply(get_irradiance, axis=1)
    df['irradiance'] = df['irradiance'].clip(lower=0)

    # Simulate Temperature (follows sun but lags slightly, plus seasonal base)
    def get_temp(row):
        hour = row['hour']
        day = row['day_of_year']
        irr = row['irradiance']

        season_base = 15 + 10 * np.sin(2 * np.pi * (day - 80) / 365)
        daily_variation = 5 * np.sin(2 * np.pi * (hour - 9) / 24)

        # Temperature rises with irradiance
        heating = 0.01 * irr

        return season_base + daily_variation + heating + np.random.normal(0, 1)

    df['temperature'] = df.apply(get_temp, axis=1)

    # Simulate Wind Speed (Random with Weibull distribution characteristic, simplified)
    df['wind_speed'] = np.random.weibull(2, n_samples) * 3

    # Calculate Power Output
    # Simple physical model: P = Efficiency * Irradiance * Area * (1 - coeff * (T_cell - 25))
    # Assuming standard panel params
    efficiency = 0.18
    area = 10 # m^2
    temp_coeff = 0.004

    # Cell temp approximation: T_cell = T_amb + (NOCT - 20)/800 * Irradiance
    # Using simplified: T_cell = T_amb + 0.025 * Irradiance
    df['cell_temp'] = df['temperature'] + 0.025 * df['irradiance']

    def get_power(row):
        irr = row['irradiance']
        t_cell = row['cell_temp']

        if irr <= 0:
            return 0.0

        base_power = efficiency * irr * area
        # Temperature loss
        temp_loss_factor = 1 - temp_coeff * (t_cell - 25)
        power = base_power * temp_loss_factor

        # Inverter efficiency and losses
        power = power * 0.95

        return max(0, power)

    df['power_output'] = df.apply(get_power, axis=1)

    # Drop intermediate columns
    df = df.drop(columns=['cell_temp', 'day_of_year'])

    return df

if __name__ == "__main__":
    print("Generating synthetic solar data...")
    data = generate_solar_data(datetime(2023, 1, 1), days=365)

    output_path = os.path.join("data", "solar_data.csv")
    os.makedirs("data", exist_ok=True)
    data.to_csv(output_path, index=False)
    print(f"Data saved to {output_path}")
    print(data.head())
