import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

def predict_location(rssi_data, model, scaler):
    rssi_data = np.array(rssi_data).reshape(1, -1)
    rssi_data = scaler.transform(rssi_data)
    predicted_coordinates = model.predict(rssi_data)
    return predicted_coordinates[0]

# 加载保存的模型
model = tf.keras.models.load_model('indoor_localization_model.h5')

# 加载保存的标准化器
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# 读取测试数据
test_data_file = 'mean_kalman_3.csv'  # 请将此替换为您的测试数据CSV文件名
test_data = pd.read_csv(test_data_file)

# 准备测试数据
X_test = test_data[['RFstar_6126','RFstar_C697','RFstar_10A6','RFstar_4DDC','RFstar_C651','RFstar_5CCE','RFstar_2684','RFstar_EB9D','RFstar_26E1','RFstar_F32C'
] ].values
y_test = test_data.iloc[:, :2].values

# 计算预测坐标
predicted_coordinates = np.array([predict_location(x, model, scaler) for x in X_test])

# 计算误差
errors = np.sqrt(np.sum((y_test - predicted_coordinates) ** 2, axis=1))
# 创建新的DataFrame存储实际坐标、预测坐标和误差
result_df = pd.DataFrame()
result_df['Point'] = np.arange(1, len(X_test) + 1)
result_df['Predicted_X'] = predicted_coordinates[:, 0]
result_df['Predicted_Y'] = predicted_coordinates[:, 1]
result_df['Actual_X'] = y_test[:, 0]
result_df['Actual_Y'] = y_test[:, 1]
result_df['Error'] = errors

# 将结果写入CSV文件
result_df.to_csv('results.csv', index=False)
# 输出实际坐标、预测坐标和误差
for i in range(len(X_test)):
    print(f"实际坐标: {y_test[i]}, 预测坐标: {predicted_coordinates[i]}, 误差: {errors[i]}")

# 输出平均误差、最大误差和最小误差
print(f"\n平均误差: {errors.mean()}, 最大误差: {errors.max()}, 最小误差: {errors.min()}")
# 绘制累计概率图
sorted_errors = np.sort(errors)
p = 1. * np.arange(len(errors)) / (len(errors) - 1)
plt.plot(sorted_errors, p)
plt.xlabel('Error')
plt.ylabel('Cumulative Probability')
plt.title('Cumulative Distribution Function of Errors')
plt.show()