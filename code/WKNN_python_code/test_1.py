import os
import pandas as pd
import numpy as np
import joblib
import math
import matplotlib.pyplot as plt

# 高斯权重函数定义（向量化版本）
def gaussian_weight(distances, sigma=1.0):
    return np.exp(-(distances ** 2) / (2 * sigma ** 2))
# 设置文件夹路径和CSV文件名
#folder_path = 'your_folder_path'  # 请将这里的文件夹路径修改为实际的文件夹路径
#csv_file = 'mean_kalman_test.csv'  # 请将这里的文件名修改为实际的CSV文件名
csv_files = ['mean_kalman_test.csv','mean_kalman_3.csv']
# 读取测试数据
data_frames = []
for file in csv_files:
    data_frames.append(pd.read_csv(file))
test_data = pd.concat(data_frames)

# 提取特征和标签
X_test = test_data.drop(columns=['X', 'Y','RFstar_F65C'])
y_test = test_data[['X', 'Y']]

# 加载保存的KNN模型
model_file = 'mean_kalman_gwknn_model.pkl'  # 请根据实际情况修改模型文件名
#model_file = 'mean_gwknn_model.pkl'
#model_file = 'mean_within_std_gwknn_model.pkl'
#model_file = 'mean_within_std_kalman_gwknn_model.pkl'
#model_file = 'mean_kalman_gwknn_model.pkl'
knn_model = joblib.load(model_file)

# 使用模型进行预测
y_pred = knn_model.predict(X_test)

# 计算预测坐标与实际坐标的误差（欧式距离）

errors = np.sqrt(np.sum((y_pred - y_test) ** 2, axis=1))

# 绘制预测坐标和实际坐标
colors = np.where(errors > 2, 'r', 'b')
plt.scatter(y_test['X'], y_test['Y'], marker='s', c=colors, label='Actual Coordinates')
#plt.scatter(y_pred[:, 0], y_pred[:, 1], marker='^', c='r', label='Predicted Coordinates')

# 添加图例
plt.legend()

# 显示图形
plt.show()
# 输出每个点的预测坐标、实际坐标和误差
result_df = pd.DataFrame(columns=['Point', 'Predicted_X', 'Predicted_Y', 'Actual_X', 'Actual_Y', 'Error'])
# 输出每个点的预测坐标、实际坐标和误差
# 输出每个点的预测坐标、实际坐标和误差
# 输出每个点的预测坐标、实际坐标和误差
result_df = pd.DataFrame(columns=['Point', 'Predicted_X', 'Predicted_Y', 'Actual_X', 'Actual_Y', 'Error'])
print("y_test shape:", y_test.shape[0])
print("y_pred shape:", y_pred.shape[0])
result_df.reset_index(drop=True, inplace=True)

# 输出每个点的预测坐标、实际坐标和误差
for i in range(len(y_test)):
    #result_df.loc[i] = [i+1, y_pred[i, 0], y_pred[i, 1], y_test.iloc[i, 0], y_test.iloc[i, 1], errors[i]]
    print(f'Point {i+1}:')
    print(f'  Predicted Coordinates: ({y_pred[i, 0]:.2f}, {y_pred[i, 1]:.2f})')
    print(f'  Actual Coordinates: ({y_test.iloc[i, 0]:.2f}, {y_test.iloc[i, 1]:.2f})')
    print(f'  Error: {errors.values[i]:.2f}')

# 保存到新的CSV文件
result_df.to_csv('results.csv', index=False)
# 计算平均误差、最大误差和最小误差
mean_error = np.mean(errors)
max_error = np.max(errors)
min_error = np.min(errors)

print(f'Mean error: {mean_error:.2f}')
print(f'Max error: {max_error:.2f}')
print(f'Min error: {min_error:.2f}')
# 绘制累计概率图
sorted_errors = np.sort(errors)
p = 1. * np.arange(len(errors)) / (len(errors) - 1)
plt.plot(sorted_errors, p)
plt.xlabel('Error')
plt.ylabel('Cumulative Probability')
plt.title('Cumulative Distribution Function of Errors')
plt.show()