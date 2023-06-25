import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm.keras import TqdmCallback
from keras.callbacks import Callback
from tqdm import tqdm
from keras.callbacks import Callback

class CustomTqdmCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % 5 == 0:
            logs = logs or {}
            progress_info = f"Epoch {epoch + 1}: "
            for metric, value in logs.items():
                progress_info += f"{metric}={value:.4f}, "
            print(progress_info)

        
# 加载CSV数据
csv_file = 'mean_within_std_kalman.csv'  # 请将此替换为您的CSV文件名
data = pd.read_csv(csv_file)

# 准备数据
X = data[['RFstar_6126','RFstar_C697','RFstar_10A6','RFstar_4DDC','RFstar_C651','RFstar_5CCE','RFstar_2684','RFstar_EB9D','RFstar_26E1','RFstar_F32C'
] ]
y = data.iloc[:, :2].values

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# 数据标准化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# 保存标准化器
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# 构建神经网络模型
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='sigmoid', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(32, activation='sigmoid'),
    tf.keras.layers.Dense(16, activation='sigmoid'),
    tf.keras.layers.Dense(2)
])

# 编译模型
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

early_stopping = tf.keras.callbacks.EarlyStopping(patience=210, restore_best_weights=True)

# 训练模型
model.fit(
    X_train,
    y_train,
    #epochs=372,
    epochs=10000,
    batch_size=8,
    validation_data=(X_test, y_test),
    verbose=0,
    callbacks=[CustomTqdmCallback(),early_stopping],
)


# 保存模型到当前文件夹
model.save('indoor_localization_model.h5')

print("模型已成功保存在当前文件夹中。")
