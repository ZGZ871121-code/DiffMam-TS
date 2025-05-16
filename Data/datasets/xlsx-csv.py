import pandas as pd

# 读取 Excel 文件
df = pd.read_excel("mahu133train.xlsx", sheet_name=None)  # 读取所有工作表

# 选择其中一个工作表并保存为 CSV
df['Sheet1'].to_csv('mahu133train.csv', index=False, encoding='utf-8')