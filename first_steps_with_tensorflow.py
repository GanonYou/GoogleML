#!/usr/bin/python
# -*- coding: UTF-8 -*-

from __future__ import print_function

import math

from IPython import display
from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
import tensorflow as tf
from tensorflow.python.data import Dataset

tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format

#通过pandas从特定网址加载数据集
california_housing_dataframe = pd.read_csv("https://download.mlcc.google.com/mledu-datasets/california_housing_train.csv", sep=",")

#对数据进行随机化处理，调用了numpy中的函数
california_housing_dataframe = california_housing_dataframe.reindex(
    np.random.permutation(california_housing_dataframe.index))

california_housing_dataframe["median_house_value"] /= 1000.0

#调用print函数才会显示内容
#print(california_housing_dataframe.describe()) 

#-------- step1:定义特征并配置特征列 ----------#
#比如提取total_rooms这一列
my_feature = california_housing_dataframe[["total_rooms"]]

#定义一个特征列，特征列仅存储对特征数据的描述，但不包含特征数据本身
feature_columns = [tf.feature_column.numeric_column("total_rooms")]

#-------- step2:定义目标 ----------#
#这里的目标显然是房价
targets =  california_housing_dataframe["median_house_value"]

#-------- step3:配置线性回归模型 ----------#
#使用梯度下降算法作为优化模型的方法，这里学习率设为0.00000001
my_optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.00000001)

#将梯度裁剪应用到优化器，梯度裁剪可确保梯度大小在训练期间不会变得过大，过大会导致梯度下降失败
my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer,5.0)

#根据定义好的feature_columns和my_optimizer来配置线性回归模型
linear_regressor = tf.estimator.LinearRegressor(
	feature_columns = feature_columns,
	optimizer = my_optimizer
)

#-------- step4:定义输入函数 ----------#
# 要将加利福尼亚州住房数据导入 LinearRegressor，我们需要定义一个输入函数，让它告诉 TensorFlow 如何对数据进行预处理，以及在模型训练期间如何批处理、随机处理和重复数据。
# 首先，我们将 Pandas 特征数据转换成 NumPy 数组字典。然后，我们可以使用 TensorFlow Dataset API 根据我们的数据构建 Dataset 对象，并将数据拆分成大小为 batch_size 的多批数据，以按照指定周期数 (num_epochs) 进行重复。
# 注意：如果将默认值 num_epochs=None 传递到 repeat()，输入数据会无限期重复。
# 然后，如果 shuffle 设置为 True，则我们会对数据进行随机处理，以便数据在训练期间以随机方式传递到模型。buffer_size 参数会指定 shuffle 将从中随机抽样的数据集的大小。
# 最后，输入函数会为该数据集构建一个迭代器，并向 LinearRegressor 返回下一批数据
def my_input_fn(features,targets,batch_size = 1,shuffle = True,num_epochs = None):
    #这里得到的features是一个dict，具体为{'total_rooms': array([ 5039.,1840., ..., 705.])}
    #其仅有一对{key，value}值，这里的value实际上是一个array对象
    features = {key:np.array(value) for key,value in dict(features).items()}
    #构造一个数据集
    ds = Dataset.from_tensor_slices((features,targets))
    ds = ds.batch(batch_size).repeat(num_epochs)
    #若shuffle为真，则对数据进行随机处理
    if shuffle:
    	ds = ds.shuffle(buffer_size = 10000)

    #返回下一批数据
    features,labels = ds.make_one_shot_iterator().get_next()
    return features,labels


#-------- step5:训练模型 ----------#
_ = linear_regressor.train(
    input_fn = lambda:my_input_fn(my_feature,targets),
    steps = 100
    )

#-------- step6:评估模型 ----------#
#为predictions定义一个输入函数
predictions_input_fn = lambda:my_input_fn(my_feature,targets,num_epochs = 1,shuffle = True)

#调用linear_regressor的predict函数
predictions = linear_regressor.predict(input_fn = predictions_input_fn)

#将predictions格式化为numpy的array
predictions = np.array([item['predictions'][0] for item in predictions])

#计算均方误差(MSE)
mean_squared_error = metrics.mean_squared_error(predictions,targets)
#由于均方误差很难解读，因此我们经常使用的是均方根误差(RMSE)，它与原目标同规模
root_mean_squared_error = math.sqrt(mean_squared_error)

print("Mean Squared Error (on training data): %.3f" % mean_squared_error)
print("Root Mean Squared Error (on training data): %.3f" % root_mean_squared_error)

#下面开始绘图，随机挑选一些样本
sample = california_housing_dataframe.sample(n = 300)

x_0 = sample["total_rooms"].min()
x_1 = sample["total_rooms"].max()

#从训练好的线性回归模型中得到的weights与bias
weights = linear_regressor.get_variable_value('linear/linear_model/total_rooms/weights')[0]
bias =  linear_regressor.get_variable_value('linear/linear_model/bias_weights')

y_0 = weights * x_0 + bias
y_1 = weights * x_1 + bias

#绘制一条直线，这就是训练好的线性回归模型
plt.plot([x_0,x_1],[y_0,y_1],c = 'r')
plt.xlabel("total_rooms")
plt.ylabel("median_house_value")

#绘制一些样本中的散点作为对照
plt.scatter(sample["total_rooms"],sample["median_house_value"])
plt.show()














