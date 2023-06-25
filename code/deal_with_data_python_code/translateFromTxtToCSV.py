import csv
from optparse import Values
import os
from itertools import zip_longest
# 创建新文件夹
if not os.path.exists('sourse'):
    os.mkdir('sourse')
device_names_list=['RFstar_6126','RFstar_C697','RFstar_F65C','RFstar_10A6','RFstar_4DDC','RFstar_C651','RFstar_5CCE','RFstar_2684','RFstar_EB9D','RFstar_26E1','RFstar_F32C'
]    
# 打开txt文件，读取内容并按行分割
for filename in os.listdir():
    if filename.endswith('.txt'):
        print(filename)
        with open(filename, 'r',encoding="utf-8" ,newline='') as f:
            content = f.read().splitlines()

            # 初始化一个空的字典，用于存储数据
            data = {}

            # 遍历每一行内容
            for line in content:
               
                # 如果当前行包含“坐标X”关键字，则将当前坐标X作为字典的键
                if "坐标X：" in line:
                    x = line.replace("坐标X：", "")
                    if x not in data:
                        data[x] = {}
                # 如果当前行包含“坐标Y”关键字，则将当前坐标Y作为字典的值
                elif "坐标Y：" in line:
                    y = line.replace("坐标Y：", "")
                    if y not in data[x]:
                        data[x][y] = {key: [] for key in device_names_list}
                # 如果当前行包含“蓝牙设备名”关键字，则将当前蓝牙设备名作为字典的key，并初始化一个空的子字典
                elif "蓝牙设备名:" in line:
                    device_name = line.replace("蓝牙设备名:", "")
                    if device_name not in data[x][y]:
                        data[x][y][device_name]=[]
                # 如果当前行包含“蓝牙信号强度RSSI”关键字，则将当前蓝牙信号强度RSSI作为子字典的值
                elif "蓝牙信号强度RSSI:" in line:
                    rssi = line.replace("蓝牙信号强度RSSI:", "")
                    data[x][y][device_name].append (rssi)
                    #print(rssi)

        # 将数据写入csv文件
        for x, ys in data.items():
            for yName,y in ys.items():
            # 构造csv文件名，由坐标X和坐标Y组成
                filename = "sourse/"+ x + "_" + yName + ".csv"
                print(filename)
                with open(filename, 'a', newline='') as f:
                    writer = csv.writer(f)
                    # 写入表头，仅在第一次写入时写表头
                    if f.tell() == 0:
                        writer.writerow(['X','Y', 'RFstar_6126','RFstar_C697','RFstar_F65C','RFstar_10A6','RFstar_4DDC','RFstar_C651','RFstar_5CCE','RFstar_2684','RFstar_EB9D','RFstar_26E1','RFstar_F32C'])
                    xToWrite = [x]
                    yToWrite = [yName]
                    # 遍历字典中的数据，将数据按行写入csv文件
                    for values in zip_longest(xToWrite,yToWrite, data[x][yName]['RFstar_6126'],data[x][yName]['RFstar_C697'],data[x][yName]['RFstar_F65C'],data[x][yName]['RFstar_10A6'],data[x][yName]['RFstar_4DDC'],data[x][yName]['RFstar_C651'],data[x][yName]['RFstar_5CCE'],data[x][yName]['RFstar_2684'],data[x][yName]['RFstar_EB9D'],data[x][yName]['RFstar_26E1'],data[x][yName]['RFstar_F32C'], fillvalue=''):
                            writer.writerow(values)
                        