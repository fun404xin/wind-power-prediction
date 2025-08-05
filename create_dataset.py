import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import torch
from joblib import dump, load
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.fftpack import fft, ifft
import statsmodels.api as sm
# 通过滑动窗口制作多步预测数据集
def create_multistep_dataset(var_data, ylabel_data, window_size, energy_threshold=0.9):
    '''
    参数:
        var_data       : 多变量输入特征 [time_len, feature_dim]
        ylabel_data    : 单变量目标序列 [time_len, 1]
        window_size    : 滑动窗口长度
        forecast_step  : 多步预测步数
    '''
    sample_features = []
    labels = []

    for i in range(len(var_data) - window_size - forecast_step + 1):
        sample_features.append(var_data[i:i + window_size, :])  # 原始输入窗口 [window_size, feature_dim]
        labels.append(ylabel_data[i + window_size:i + window_size + forecast_step, :])
    sample_features = torch.tensor(np.array(sample_features)).float()
    labels = torch.tensor(np.array(labels)).float()
    return sample_features, labels

 # 制作多步预测数据集
def make_dataset(var_data,ylabel_data,window_size,split_rate):
  '''
      参数:
      OTddata    : 原始数据集
      window_size      : 样本窗口大小
      forecast_step    : 多步预测
      split_rate       : 数据集划分比例

      返回:
      train_set: 训练集数据
      train_label: 训练集标签
      test_set: 测试集数据
      test_label: 测试集标签
  '''
  # 第一步，划分数据集
  # 一维序列数组
  sample_len = len(var_data)  # 样本总长度
  train_len = int(sample_len * split_rate[0])  # 向下取整
  val_len = int(sample_len * split_rate[1])
  train_var = var_data[:train_len, :]  # 训练集
  val_var = var_data[train_len:train_len+val_len, :]
  test_var = var_data[train_len+val_len:, :]  # 测试集
  train_y = ylabel_data[:train_len]  # 训练集
  val_y = ylabel_data[train_len:train_len+val_len, :]
  test_y = ylabel_data[train_len+val_len:, :]  # 测试集
  # 第二步，制作数据集标签  滑动窗口
  train_set, train_label = create_multistep_dataset(train_var,train_y,window_size)
  val_set, val_label = create_multistep_dataset(val_var,val_y,window_size)
  test_set, test_label = create_multistep_dataset(test_var,test_y,window_size)
  return train_set, train_label, val_set, val_label, test_set, test_label
if __name__ == '__main__':
    matplotlib.rc("font", family='Microsoft YaHei')
    # 读取数据
    original_data = pd.read_csv('dataset/Yalova.csv')
    # original_data = pd.read_excel('dataset/wind power.xlsx')
    # original_data.drop(['Tmstamp'],axis=1, inplace=True)
    print(original_data.shape)
    original_data.head()

    plt.figure(figsize=(15, 5), dpi=100)
    plt.grid(True)
    plt.plot(original_data['LV ActivePower (kW)'], color='green')
    plt.show()

    original_data = original_data[['LV ActivePower (kW)','Wind Speed (m/s)',
                                   'Theoretical_Power_Curve (KWh)']]
    # original_data = original_data[['Wspd','Patv']]
    original_data = original_data.interpolate(method='polynomial',order=2)
    # original_data.dropna()
    original_data['LV ActivePower (kW)'] = original_data['LV ActivePower (kW)'].apply(lambda x:max(0,x))
    print(original_data.shape)
    # var_data = original_data.drop(columns=['LV ActivePower (kW)'])
    var_data = original_data
    ylabel_data = original_data[['LV ActivePower (kW)']]
    scaler = StandardScaler()
    var_data = scaler.fit_transform(var_data)
    ylabel_data = scaler.fit_transform(ylabel_data)
    # 保存 归一化 模型
    dump(scaler, 'scaler')
     # 定义序列长度和预测步数
     # 定义窗口大小  ： 用过去 18 个步长 ，预测未来 6 个步长
    window_size = 18
     # 多步预测
    forecast_step = 6
     # 数据集划分比例
    split_rate = [0.7, 0.15, 0.15]

     # 数据集制作
    train_xdata, train_ylabel, val_xdata, val_ylabel, test_xdata, test_ylabel = make_dataset(var_data,ylabel_data,window_size,split_rate)
     # 保存数据
    dump(train_xdata, 'train_xdata')
    dump(val_xdata, 'val_xdata')
    dump(test_xdata, 'test_xdata')
    dump(train_ylabel, 'train_ylabel')
    dump(val_ylabel, 'val_ylabel')
    dump(test_ylabel, 'test_ylabel')
    print('train_xdata.shape',train_xdata.shape)
    print('train_ydata.shape', train_ylabel.shape)
    print('val_xdata.shape',val_xdata.shape)
    print('val_ydata.shape', val_ylabel.shape)
    print('test_xdata.shape',test_xdata.shape)
    print('test_ydata.shape',test_ylabel.shape)
