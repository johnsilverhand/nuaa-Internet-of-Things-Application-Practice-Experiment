import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata

# 获取当前文件夹路径
current_dir = os.getcwd()

# 拼接子文件夹sourse的路径
sourse_dir = os.path.join(current_dir, 'sourse')
# 获取所有csv文件的文件名
file_list = [f for f in os.listdir(sourse_dir) if f.endswith(".csv")]

# 存储每个csv文件的数据
data_list = []

# 读取所有csv文件的数据
for file in file_list:
    df = pd.read_csv(os.path.join(sourse_dir, file))
    data_list.append(df)

# 获取所有csv文件相同表头的列名
cols = set(data_list[0].columns).intersection(*[set(df.columns) for df in data_list])
# 计算子图的行数和列数
nrows = (len(cols) + 3) // 4
ncols = 4

fig = plt.figure(figsize=(25, 15))

# 循环绘制每个子图
for i, col in enumerate(cols):
    if col == 'X' or col == 'Y':
        continue
    ax = fig.add_subplot(nrows, ncols, i+1, projection='3d')
    ax.set_title(col)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Average')
    
    # 存储每个数据集中所有点的x、y、z值
    x_list, y_list, z_list = [], [], []
    for df in data_list:
        #mu, std = df[col].mean(), df[col].std(ddof=1)
        #mask = (df[col] >= mu - std) & (df[col] <= mu + std)
         
        df.fillna(df.mean(), inplace=True)
        x = df.iloc[:, 0]
        y = df.iloc[:, 1]
        #z = df[col][mask].mean()
        z = df[col].mean() # 使用平均值作为z值
        #df.fillna(df[col][mask].mean(), inplace=True)
        x_list.extend(x)
        y_list.extend(y)
        z_list.extend([z]*len(x)) # 扩展z值的长度与x相同
        

    # 定义网格的大小和范围
    xi = np.linspace(min(x_list), max(x_list), 50)
    yi = np.linspace(min(y_list), max(y_list), 50)
    zi = griddata((x_list, y_list), z_list, (xi[None, :], yi[:, None]), method='cubic')

    x_, y_ = np.meshgrid(xi, yi)
    # 计算颜色值
    color = (zi - zi.min()) / (zi.max() - zi.min())
    # 绘制平滑曲面
    ax.plot_surface(x_, y_, zi, cmap='coolwarm', alpha=0.7)
    '''
    # 循环绘制所有数据集中的点，并设置颜色
    for df in data_list:
        x = df.iloc[:, 0]
        y = df.iloc[:, 1]
        z = df[col].mean() # 使用算术平均值作为z值
        color_value = (z - zi.min()) / (zi.max() - zi.min()) # 计算颜色值
        ax.scatter(x, y, z, color=plt.cm.RdBu(color_value))
    # 设置z轴范围
    '''
    ax.set_zlim(-90, -50)   
    
plt.tight_layout()

# 显示图像
plt.savefig('3d_surface_average_plots2.png')
#plt.show()