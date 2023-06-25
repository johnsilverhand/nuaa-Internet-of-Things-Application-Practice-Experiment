from cmath import exp
import csv
import pandas as pd
import numpy as np
import os
from scipy.stats import norm
from pykalman import KalmanFilter
if not os.path.exists('sourse'):
    os.mkdir('sourse')
if not os.path.exists('csv_output'):
    os.mkdir('csv_output')
# 获取当前文件夹路径
current_dir = os.getcwd()

# 拼接子文件夹sourse的路径
sourse_dir = os.path.join(current_dir, 'sourse')
#目标文件名
mean_filename = 'mean.csv'
mean_within_std_filename = 'mean_within_std.csv'
mean_kalmam_filename = 'mean_kalman.csv'
mean_within_std_kalman_filename = 'mean_within_std_kalman.csv'
#表头
header = ['X','Y', 'RFstar_6126','RFstar_C697','RFstar_F65C','RFstar_10A6','RFstar_4DDC','RFstar_C651','RFstar_5CCE','RFstar_2684','RFstar_EB9D','RFstar_26E1','RFstar_F32C']
for filename in os.listdir(sourse_dir):
    if filename.endswith('.csv'):
        # 读取CSV文件
        data = pd.read_csv('sourse/'+filename)

        # 提取第三列到第十二列的数据
        df = data.iloc[:, 2:13]
        x = data.iloc[0, 0]
        y = data.iloc[0, 1]
        line_mean = {}
        line_mean_within_std = {}
        line_mean_kalman = {}
        line_mean_within_std_kalman = {}
        for i, col in enumerate(df.columns):
            #均值滤波和高斯滤波相关变量
            # 计算均值和标准差
            mu, std = df[col].mean(), df[col].std(ddof=1)
            # 计算均值在标准差范围内的数据的均值
            mask = (df[col] >= mu - std) & (df[col] <= mu + std)
            # 计算均值在标准差范围内的数据的均值
            mean_within_std = df[col][mask].mean()

            # 使用 KalmanFilter 对数据进行滤波
            kf = KalmanFilter(initial_state_mean=df[col].values[0],initial_state_covariance=10,observation_covariance=4,transition_covariance=exp(-4))
            filtered_data = kf.filter(df[col].values)[0]
            mu_kalmam = np.nanmean(filtered_data)
            std_kalman = np.nanstd(filtered_data)
            mask_kalman = (filtered_data >= mu_kalmam - 1.2*std_kalman) & (filtered_data <= mu_kalmam + 1.2*std_kalman)
            mean_within_std_kalman = np.nanmean(filtered_data[mask_kalman])
            # 将数据存入字典
            line_mean[col] = mu
            line_mean_within_std[col] = mean_within_std
            line_mean_kalman[col] = mu_kalmam
            line_mean_within_std_kalman[col] = mean_within_std_kalman
        #写入文件
        with open(mean_filename, 'a', newline='') as f:
            writer = csv.writer(f)
            # 写入表头，仅在第一次写入时写表头
            if f.tell() == 0:
                writer.writerow(header)
                # 按照header的顺序遍历字典中的数据，将数据按行写入csv文件     
            row = [x, y]
            for col in header[2:]:
                row.append(line_mean[col])
            writer.writerow(row)
        with open(mean_within_std_filename, 'a', newline='') as f:
            writer = csv.writer(f)
            if f.tell() == 0:
                writer.writerow(header)
            row = [x, y]
            for col in header[2:]:
                row.append(line_mean_within_std[col])
            writer.writerow(row)
        with open(mean_kalmam_filename, 'a', newline='') as f:
            writer = csv.writer(f)
            if f.tell() == 0:
                writer.writerow(header)
            row = [x, y]
            for col in header[2:]:
                row.append(line_mean_kalman[col])
            writer.writerow(row)
        with open(mean_within_std_kalman_filename, 'a', newline='') as f:
            writer = csv.writer(f)
            if f.tell() == 0:
                writer.writerow(header)
            row = [x, y]
            for col in header[2:]:
                row.append(line_mean_within_std_kalman[col])
            writer.writerow(row)       