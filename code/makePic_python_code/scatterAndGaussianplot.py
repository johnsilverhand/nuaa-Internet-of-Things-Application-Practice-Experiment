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

        # 绘制散点图和高斯分布的概率密度函数
        for i, col in enumerate(df.columns):
            if col == 'X' or col == 'Y':
                continue
            ax = axs[i]
            mu, std = df[col].mean(), df[col].std(ddof=1)
            xmin, xmax = df[col].min(), df[col].max()
            x = np.linspace(xmin, xmax, 100)
            p = norm.pdf(x, mu, std)
            # 将每个数据点的 y 坐标微小地浮动 norm.pdf(df[col], mu, std)
            y_jittered = norm.pdf(df[col], mu, std) -np.abs( np.random.normal(0, 0.01, size=len(df)))
            ax.plot(x, p, label=f'{col} ({mu:.2f}, {std:.2f})', color=colors[i])
            ax.fill_between(x, 0, p, where=(x >= mu - std) & (x <= mu + std), alpha=0.3,facecolor=colors[i])
            ax.scatter(df[col], y_jittered , alpha=0.6, label=col, color=colors[i])

            # 设置子图的标题、轴标签和图例
            ax.set_title(col)
            ax.set_xlabel('RSSI')
            ax.set_ylabel('probability')
            ax.legend()

        # 将子图合并为一张大图
        plt.suptitle(filename.split(".csv")[0])
        plt.tight_layout()

        # 保存和显示图像在scatterAndGaussian子文件夹中，请提前创建好
        plt.savefig("scatterAndGaussian/"+filename.split(".csv")[0]+'scatterAndGaussian_subplots.png')
        #plt.show()
