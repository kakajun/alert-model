import torch
import joblib
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from datetime import datetime
from model import SolarPredictor
from data_generator import generate_solar_data


class SolarModel:
    def __init__(self, model_dir="models"):
        self.model_dir = model_dir
        self.model = None
        self.scaler_X = None
        self.scaler_y = None
        self._load_model()

    def _load_model(self):
        model_path = os.path.join(self.model_dir, "solar_model.pth")
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model not found at {model_path}. Please run train.py first.")

        self.scaler_X = joblib.load(
            os.path.join(self.model_dir, "scaler_X.pkl"))
        self.scaler_y = joblib.load(
            os.path.join(self.model_dir, "scaler_y.pkl"))

        # 我们需要知道输入维度，基于训练脚本设定为 5
        self.model = SolarPredictor(input_dim=5)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

    def predict(self, hour, month, temperature, irradiance, wind_speed):
        # 准备输入
        input_data = np.array(
            [[hour, month, temperature, irradiance, wind_speed]])
        input_scaled = self.scaler_X.transform(input_data)
        input_tensor = torch.FloatTensor(input_scaled)

        # 预测
        with torch.no_grad():
            prediction_scaled = self.model(input_tensor)
            prediction = self.scaler_y.inverse_transform(
                prediction_scaled.numpy())

        return max(0.0, prediction[0][0])

    def predict_batch(self, df):
        """
        批量预测
        df: 包含 ['hour', 'month', 'temperature', 'irradiance', 'wind_speed'] 列的 DataFrame
        """
        feature_cols = ['hour', 'month',
                        'temperature', 'irradiance', 'wind_speed']
        input_data = df[feature_cols].values
        input_scaled = self.scaler_X.transform(input_data)
        input_tensor = torch.FloatTensor(input_scaled)

        with torch.no_grad():
            prediction_scaled = self.model(input_tensor)
            prediction = self.scaler_y.inverse_transform(
                prediction_scaled.numpy())

        # 将小于0的预测值置为0
        return np.maximum(0.0, prediction.flatten())


def main():
    print("Solar Power Prediction Demo - Daily Simulation")
    print("--------------------------------------------")

    # 1. 生成一整天的模拟数据 (例如 6月21日夏至)
    print("Generating simulated data for a full day...")
    test_date = datetime(2023, 6, 21)
    df = generate_solar_data(test_date, days=1, interval_minutes=60)

    # 2. 加载模型并预测
    try:
        predictor = SolarModel()
        predicted_power = predictor.predict_batch(df)
        df['predicted_power'] = predicted_power
    except Exception as e:
        print(f"Error loading model or predicting: {e}")
        return

    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei',
                                       'Microsoft YaHei', 'Arial Unicode MS']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

    # 3. 绘制结果
    print("Plotting results...")
    plt.figure(figsize=(12, 6))

    # 绘制实际模拟值（基准值）
    plt.plot(df['hour'], df['power_output'], 'b-o',
             label='模拟真实值 (物理模型)', linewidth=2, alpha=0.7)

    # 绘制预测值
    plt.plot(df['hour'], df['predicted_power'],
             'r--x', label='神经网络预测值', linewidth=2)

    plt.title(f"太阳能发电预测 vs 真实值 ({test_date.strftime('%Y-%m-%d')})")
    plt.xlabel("时间 (小时)")
    plt.ylabel("功率输出 (kW)")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.xticks(df['hour'])

    # 保存图表
    output_img = "prediction_plot.png"
    plt.savefig(output_img)
    print(f"Plot saved to {output_img}")

    # 4. 另外绘制天气条件图表 (温度, 辐照度, 风速)
    print("Plotting weather conditions...")
    plt.figure(figsize=(12, 10))

    # 子图1: 辐照度
    plt.subplot(3, 1, 1)
    plt.plot(df['hour'], df['irradiance'], 'orange',
             marker='o', label='辐照度 (W/m²)')
    plt.title(f"天气条件 ({test_date.strftime('%Y-%m-%d')})")
    plt.ylabel("辐照度 (W/m²)")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='upper right')
    plt.xticks(df['hour'], [])  # 隐藏x轴标签

    # 子图2: 温度
    plt.subplot(3, 1, 2)
    plt.plot(df['hour'], df['temperature'], 'r', marker='s', label='温度 (°C)')
    plt.ylabel("温度 (°C)")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='upper right')
    plt.xticks(df['hour'], [])  # 隐藏x轴标签

    # 子图3: 风速
    plt.subplot(3, 1, 3)
    plt.plot(df['hour'], df['wind_speed'], 'g', marker='^', label='风速 (m/s)')
    plt.ylabel("风速 (m/s)")
    plt.xlabel("时间 (小时)")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='upper right')
    plt.xticks(df['hour'])

    plt.tight_layout()

    # 保存天气图表
    weather_img = "weather_plot.png"
    plt.savefig(weather_img)
    print(f"Weather plot saved to {weather_img}")

    # 尝试显示（如果在支持的显示环境中）
    try:
        plt.show()
    except:
        pass

    # 打印对比数据
    print("\nHourly Comparison:")
    print(f"{'Hour':<6} | {'Ground Truth (kW)':<18} | {'Prediction (kW)':<18} | {'Diff':<10}")
    print("-" * 60)
    for i in range(len(df)):
        row = df.iloc[i]
        diff = row['predicted_power'] - row['power_output']
        print(
            f"{int(row['hour']):<6} | {row['power_output']:<18.2f} | {row['predicted_power']:<18.2f} | {diff:<10.2f}")


if __name__ == "__main__":
    main()
