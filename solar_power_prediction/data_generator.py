import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta


def generate_solar_data(start_date, days=365, interval_minutes=60):
    """
    基于物理原理生成合成太阳能发电数据。
    """
    date_range = [start_date + timedelta(minutes=i*interval_minutes)
                  for i in range(days * 24 * (60 // interval_minutes))]

    n_samples = len(date_range)

    df = pd.DataFrame({'timestamp': date_range})
    df['hour'] = df['timestamp'].dt.hour
    df['month'] = df['timestamp'].dt.month
    df['day_of_year'] = df['timestamp'].dt.dayofyear

    # 模拟辐照度（白天的钟形曲线，经季节性调整）
    # 中午达到峰值（12点），夜间为零
    # 季节性：夏季较高（北半球约6月），冬季较低

    def get_irradiance(row):
        hour = row['hour']
        day = row['day_of_year']

        # 季节因子（年中达到峰值的简单正弦波）
        season_factor = 1 + 0.4 * np.sin(2 * np.pi * (day - 80) / 365)

        # 日模式（类高斯分布）
        if 6 <= hour <= 18:
            # 白天归一化小时数从 -1 到 1
            h_norm = (hour - 12) / 6
            daily_factor = np.exp(-2 * h_norm**2)
            # 添加随机云层遮挡噪声
            noise = np.random.uniform(0.8, 1.0) if np.random.random(
            ) > 0.2 else np.random.uniform(0.1, 0.6)
            return 1000 * daily_factor * season_factor * noise
        else:
            return 0.0

    df['irradiance'] = df.apply(get_irradiance, axis=1)
    df['irradiance'] = df['irradiance'].clip(lower=0)

    # 模拟温度（随太阳变化但略有滞后，加上季节性基准）
    def get_temp(row):
        hour = row['hour']
        day = row['day_of_year']
        irr = row['irradiance']

        season_base = 15 + 10 * np.sin(2 * np.pi * (day - 80) / 365)
        daily_variation = 5 * np.sin(2 * np.pi * (hour - 9) / 24)

        # 温度随辐照度升高
        heating = 0.01 * irr

        return season_base + daily_variation + heating + np.random.normal(0, 1)

    df['temperature'] = df.apply(get_temp, axis=1)

    # 模拟风速（具有威布尔分布特征的随机数，简化版）
    df['wind_speed'] = np.random.weibull(2, n_samples) * 3

    # 计算功率输出
    # 简单物理模型：P = 效率 * 辐照度 * 面积 * (1 - 系数 * (电池温度 - 25))
    # 假设标准电池板参数
    efficiency = 0.18
    area = 10  # m^2
    temp_coeff = 0.004

    # 电池温度近似值：T_cell = 环境温度 + (NOCT - 20)/800 * 辐照度
    # 使用简化版：T_cell = 环境温度 + 0.025 * 辐照度
    df['cell_temp'] = df['temperature'] + 0.025 * df['irradiance']

    def get_power(row):
        irr = row['irradiance']
        t_cell = row['cell_temp']

        if irr <= 0:
            return 0.0

        base_power = efficiency * irr * area
        # 温度损耗
        temp_loss_factor = 1 - temp_coeff * (t_cell - 25)
        power = base_power * temp_loss_factor

        # 逆变器效率和损耗 (添加一些随机波动，模拟设备状态变化)
        inverter_efficiency = 0.95 + np.random.uniform(-0.05, 0.02)
        power = power * inverter_efficiency

        # 添加额外的随机噪声 (模拟测量误差、灰尘遮挡等未捕获因素)
        # 噪声比例在 -10% 到 +5% 之间
        random_noise = np.random.uniform(0.90, 1.05)
        power = power * random_noise

        return max(0, power)

    df['power_output'] = df.apply(get_power, axis=1)

    # 删除中间列
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
