import csv
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.stats import norm

if not os.path.exists('sourse'):
    os.mkdir('sourse')
# 获取当前文件夹路径
current_dir = os.getcwd()

# 拼接子文件夹sourse的路径
sourse_dir = os.path.join(current_dir, 'sourse')
for filename in os.listdir(sourse_dir):
    if filename.endswith('.csv'):
        # 读取CSV文件
        #print(filename)
        data = pd.read_csv('sourse/'+filename)

        # 提取第三列到第十二列的数据
        df = data.iloc[:, 2:13]

        # 计算子图的行数和列数
        nrows = (df.shape[1] + 1) // 2
        ncols = 2

        # 创建画布和子图
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, nrows*5))
        axs = axs.flatten()

        # 定义颜色映射
        cmap = plt.get_cmap('viridis')
        colors = [cmap(i) for i in np.linspace(0, 1, df.shape[1])]

        # 绘制折线图和平均值线
        for i, col in enumerate(df.columns):
            ax = axs[i]
            mu, std = df[col].mean(), df[col].std(ddof=1)
            mask = (df[col] >= mu - std) & (df[col] <= mu + std)
            mean_within_std = df[col][mask].mean()
            #print(GaussianFilter)
            ax.plot(df[col], marker='o')
            ax.axhline(mu, color='red', linestyle='--',label=f'mean of data')
            ax.axhline(mean_within_std, linestyle='--', color='blue', label=f'mean of data between mu-std and mu+std')
        
            ax.legend()
            # 设置子图的标题、轴标签和图例
            ax.set_title(col)
            ax.set_xlabel('Count')
            ax.set_ylabel('RSSI')

        # 将子图合并为一张大图
        #plt.legend()
        plt.suptitle(filename.split(".csv")[0])
        plt.tight_layout()

        # 保存和显示图像在scatterAndGaussian子文件夹中，请提前创建好
        plt.savefig("plotOfEveryPoint/"+filename.split(".csv")[0]+'plot.png')
        #plt.show()

