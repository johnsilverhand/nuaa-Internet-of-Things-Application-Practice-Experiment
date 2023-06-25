# 21级南航物联网应用实践实验（基于无线指纹库的蓝牙室内定位实验）
- [Badge](#badge)
- [项目背景](#项目背景)
- [使用前准备](#使用前准备)
- [原始实验报告](#原始实验报告)
- [采集数据APP](#采集数据app)
- [数据](#数据)
- [转换数据代码](#转换数据代码)
- [制图代码](#制图代码)
- [制作指纹库代码](#制作指纹库代码)
- [定位代码](#定位代码)
- [开源协议](#开源协议)
## Badge
![LICENSE](https://img.shields.io/badge/license-GPL-blue)
![Python](https://img.shields.io/badge/Python-3.10-blue)


## 项目背景

本项目是笔者在南航本科21级物联网应用实践课程实验中，在小组实验中关于笔者参与部分的相关资料（代码，报告初稿，图表，数据等）。供笔者和有需要人士查阅浏览与参考。

请注意，使用代码时需要对一些部分进行修改以适应自己的情况，请自行浏览相关代码。笔者不再赘述。

如果有疑问和问题也可以联系笔者。可通过电子邮箱wangzichen16@nuaa.edu.cn
或者在issue提出。但笔者不一定都能尽快回答。

## 使用前准备
使用前请先浏览原始实验报告。
运行代码前请确保代码所需要或者所要处理的数据文件在同一文件夹中。需要自行整理至同一文件夹中。

当前项目代码使用了Python语言编写，您需要确保已经安装Python及其相关环境。（笔者使用的Python版本为3.10）

您在运行相关代码时需要确保安装了所必需的Python库

您可在运行所要运行的代码前浏览代码开头 **import**部分来校对自己是否已经安装了需要的Python库。若缺少，则请安装。


## 原始实验报告

原始实验报告为笔者使用latex编写的实验报告稿子，并非为最后的实验报告成稿。成稿和一些其他部分由其他小组成员完成。

## 采集数据APP

本项目使用任课老师提供的APP采集接收到的传感器的无线信号强度。APP仅可在安卓系统上运行。使用该APP记录到的数据将写入在一个以记录时的日期作为标识的txt文件中。

## 数据
相关数据存放在`data`文件夹中
## 转换数据代码

该部分代码存储在 `code`文件夹中的`deal_with_data_python_code`子文件夹中，文件名为**translateFromTxtToCSV.py**
使用采集数据APP得到的txt文件不方便处理数据。该部分代码意在将txt文件的数据转换为csv表格文件。运行该文件代码将会得到多个以坐标点坐标为标识的csv文件，存储在source文件夹中。

## 制图代码
该部分代码存储在 `code`文件夹中的`deal_with_data_python_code`子文件夹中，
该部分代码用于制作实验报告中的相关图片用于方便解释说明。代码及图片效果见实验报告正文部分和实验报告附录部分。

## 制作指纹库代码
该部分代码存储在 `code`文件夹中的`makePic_python_code`子文件夹中
该部分代码将`sourse`文件夹中的多个csv表格文件经过数据预处理后合并为一个csv表格文件，该文件就是定位算法中所要使用的指纹库。

## 定位代码

### 神经网络法代码
该部分代码存储在 `code`文件夹中的`neural_network_python_code`子文件夹中
运行**indoor_localization.py**，将会训练一个定位用神经网络模型。
运行**load_and_predict.py**，将会加载使用运行**indoor_localization.py**得到的定位用神经网络模型。
### WKNN法代码
该部分代码存储在 `code`文件夹中的`WKNN_python_code`子文件夹中
运行**KNNtry.py**，将会训练一个定位用的WKNN模型。
运行**test_1.py**，将会加载使用运行**KNNtry.py**得到的WKNN模型。


## 开源协议

本项目使用了GPL开源协议
