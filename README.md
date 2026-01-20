# 太阳能光伏发电预测演示 (Solar Photovoltaic Power Prediction Demo)

这是一个基于 Elmoaqet 和 Karasneh 的论文 "Predicting Solar Photovoltaic Power Production using Neural Networks" 概念的演示项目。

该项目使用多层感知机 (MLP) 神经网络，根据天气条件预测太阳能发电量。

## 项目结构

- `data_generator.py`: 生成基于物理公式的模拟数据集（包含辐照度、温度、风速等）。
- `model.py`: 定义 PyTorch 神经网络模型结构。
- `train.py`: 使用生成的数据训练模型。
- `predict.py`: 演示如何加载模型并进行预测。

## 如何运行

1.  **安装依赖**:
    确保你已经安装了 Python 环境。然后安装必要的库：
    ```bash
    pip install -r requirements.txt
    ```

2.  **生成数据**:
    运行以下命令生成模拟数据：
    ```bash
    python solar_power_prediction/data_generator.py
    ```
    这将在 `data/` 目录下生成 `solar_data.csv` 文件。

3.  **训练模型**:
    运行训练脚本：
    ```bash
    python solar_power_prediction/train.py
    ```
    训练完成后，模型和缩放器将保存到 `models/` 目录。

4.  **运行预测**:
    运行预测演示脚本：
    ```bash
    python solar_power_prediction/predict.py
    ```
    脚本将输出几种不同天气条件下的预测功率。

## 模型说明

- **输入特征**:
  - `hour`: 小时 (0-23)
  - `month`: 月份 (1-12)
  - `temperature`: 环境温度 (C)
  - `irradiance`: 太阳辐照度 (W/m^2)
  - `wind_speed`: 风速 (m/s)

- **输出**:
  - `power_output`: 预测的发电功率 (kW)
