import numpy as np
import pandas as pd

# 加载 .npy 文件
data = np.load("Data/mahu133train/ddpm_fake_mahu1.npy")

# 确认数据形状
num_samples, seq_length, feature_dim = data.shape
print(f"Data shape: {data.shape}")

# 初始化存储结构
all_data = []

for i, sample in enumerate(data):
    # 转换单个样本为 DataFrame
    df = pd.DataFrame(sample, columns=[f"Feature_{j+1}" for j in range(feature_dim)])
    df['Sample'] = i + 1  # 添加样本编号
    df['Time_Step'] = range(1, seq_length + 1)  # 添加时间步编号
    all_data.append(df)

# 合并所有样本
final_df = pd.concat(all_data, ignore_index=True)

# 调整列顺序：样本编号 -> 时间步 -> 特征
final_df = final_df[['Sample', 'Time_Step'] + [f"Feature_{j+1}" for j in range(feature_dim)]]

# 保存为 Excel 文件
output_file = 'Data/mahu133train/Fake133train.xlsx'
final_df.to_excel(output_file, index=False)

print(f"Data saved to {output_file}")