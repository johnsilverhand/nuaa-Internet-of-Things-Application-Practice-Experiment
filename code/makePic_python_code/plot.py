import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
for filename in os.listdir():
    if filename.endswith('.csv'):
        # 读取数据
        df = pd.read_csv(filename)
        data = df.iloc[:, 2:13]
        # 提取数据
        x = df['X'].to_numpy()
        y = df['Y'].to_numpy()
        z = df.drop(['X', 'Y'], axis=1).to_numpy()
        # 生成网格数据
        xi = np.linspace(min(x), max(x), 50)
        yi = np.linspace(min(y), max(y), 50)
        X, Y = np.meshgrid(xi, yi)

        # 创建画布和子图
        fig = plt.figure(figsize=(30, 30),dpi=150)
        for i, col in enumerate(data.columns):
            ax = fig.add_subplot(4, 4, i+1, projection='3d')
            z_ = griddata((x, y), data[col], (xi[None, :], yi[:, None]), method='cubic')
            ax.plot_surface(X, Y, z_, cmap='coolwarm',alpha = 0.8)
            ax.set_title('{}'.format(col))

            # 添加坐标轴标签
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('RSSI')
        plt.suptitle(filename.split('.csv')[0],fontdict={'size': 100})
        plt.tight_layout()

        # 显示图像
        plt.savefig('3d_surface_'+filename.split('.csv')[0]+'_plots.png')
