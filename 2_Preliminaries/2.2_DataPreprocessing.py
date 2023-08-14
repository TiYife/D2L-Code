import os

os.makedirs(os.path.join("..", "data"), exist_ok=True)
data_file = os.path.join('..','data','house.csv')
with open(data_file, 'w', encoding='utf-8') as f:
    f.write('NumRooms,Alley,Price\n')  # 列名
    f.write('NA,Pave,127500\n')  # 每行表示一个数据样本
    f.write('2,NA,106000\n')
    f.write('4,NA,178100\n')
    f.write('NA,NA,140000\n')

import pandas as pd

data = pd.read_csv(data_file)
print(data)
print(data.dtypes)

inputs, outputs = data.iloc[:, 0:2], data.iloc[:, 2]
print(inputs)

inputs = inputs.fillna(inputs.mean(numeric_only=True))
print(inputs)

# inputs = pd.get_dummies(inputs, dummy_na=True, dtype=int)
inputs = pd.get_dummies(inputs, dummy_na=True)
print(inputs)
print(inputs.dtypes)

inputs = inputs.astype(int)
print(inputs)
print(inputs.dtypes)

import torch

X = torch.tensor(inputs.values)
y = torch.tensor(outputs.values)
print(X)
print(y)

# ---------------------------------------------------------------------------------------------------------------------
os.makedirs(os.path.join('..', 'data'), exist_ok=True)
data_file2 = os.path.join('..', 'data', 'house_tiny.csv')
with open(data_file2, 'w') as f:
    f.write('name,age,grade,number\n')  # 列名
    f.write('li,12,NaN,123\n')  # 每行表示一个数据样本
    f.write('wang,11,NaN,234\n')
    f.write('liao,NaN,NaN,NaN\n')
    f.write('zhou,15,9,NaN\n')

data2 = pd.read_csv(data_file2)
print(data2)

nan_num = data2.isnull().sum()
print(nan_num)

data2 = data2.drop(columns=[nan_num.idxmax()])
print(data2)