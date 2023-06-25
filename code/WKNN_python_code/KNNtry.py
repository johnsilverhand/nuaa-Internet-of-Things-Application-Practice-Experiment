import os
import pandas as pd
import numpy as np
import math
from sklearn.neighbors import KNeighborsRegressor
import joblib

# 列出文件夹中的所有CSV文件
#folder_path = ''  # 请将这里的文件夹路径修改为实际的文件夹路径
csv_files = ['mean_kalman.csv']
# 计算高斯权重
def gaussian_weight(distance, sigma=1.0):
    return math.exp(-(distance ** 2) / (2 * sigma ** 2))

# 构建KNN回归模型，使用高斯权重
# 遍历每个CSV文件
for csv_file in csv_files:
    # 读取CSV文件
    data = pd.read_csv(csv_file)

    # 提取特征和标签
    X = data.drop(columns=['X', 'Y','RFstar_F65C'])
    y = data[['X', 'Y']]

    # 选择合适的k值，这里我们假设k=10
    k = 4

    # 使用k值训练KNN模型
    knn = KNeighborsRegressor(n_neighbors=k, weights='distance')
    #knn = KNeighborsRegressor(n_neighbors=k, weights=gaussian_weight)
    knn.fit(X, y)

    # 保存KNN模型到当前文件夹
    model_file = csv_file.split('.csv')[0]+'_gwknn_model.pkl'
    joblib.dump(knn, model_file)

    print(f'KNN model for {csv_file} trained with k={k} and saved to {model_file}.')