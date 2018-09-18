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
print(california_housing_dataframe.describe()) 

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




#---------- 教程提供了一个封装上述代码的函数，通过不同的参数调用来了解不同的效果 ------------
def train_model(learning_rate, steps, batch_size, input_feature="total_rooms"):
  """Trains a linear regression model of one feature.
  
  Args:
    learning_rate: A `float`, the learning rate.
    steps: A non-zero `int`, the total number of training steps. A training step
      consists of a forward and backward pass using a single batch.
    batch_size: A non-zero `int`, the batch size.
    input_feature: A `string` specifying a column from `california_housing_dataframe`
      to use as input feature.
  """
  
  periods = 10
  steps_per_period = steps / periods

  my_feature = input_feature
  my_feature_data = california_housing_dataframe[[my_feature]]
  my_label = "median_house_value"
  targets = california_housing_dataframe[my_label]

  # Create feature columns.
  feature_columns = [tf.feature_column.numeric_column(my_feature)]
  
  # Create input functions.
  training_input_fn = lambda:my_input_fn(my_feature_data, targets, batch_size=batch_size)
  prediction_input_fn = lambda: my_input_fn(my_feature_data, targets, num_epochs=1, shuffle=False)
  
  # Create a linear regressor object.
  my_optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
  my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
  linear_regressor = tf.estimator.LinearRegressor(
      feature_columns=feature_columns,
      optimizer=my_optimizer
  )

  # Set up to plot the state of our model's line each period.
  plt.figure(figsize=(15, 6))
  plt.subplot(1, 2, 1)
  plt.title("Learned Line by Period")
  plt.ylabel(my_label)
  plt.xlabel(my_feature)
  sample = california_housing_dataframe.sample(n=300)
  plt.scatter(sample[my_feature], sample[my_label])
  colors = [cm.coolwarm(x) for x in np.linspace(-1, 1, periods)]

  # Train the model, but do so inside a loop so that we can periodically assess
  # loss metrics.
  print("Training model...")
  print("RMSE (on training data):")
  root_mean_squared_errors = []
  for period in range (0, periods):
    # Train the model, starting from the prior state.
    linear_regressor.train(
        input_fn=training_input_fn,
        steps=steps_per_period
    )
    # Take a break and compute predictions.
    predictions = linear_regressor.predict(input_fn=prediction_input_fn)
    predictions = np.array([item['predictions'][0] for item in predictions])
    
    # Compute loss.
    root_mean_squared_error = math.sqrt(
        metrics.mean_squared_error(predictions, targets))
    # Occasionally print the current loss.
    print("  period %02d : %0.2f" % (period, root_mean_squared_error))
    # Add the loss metrics from this period to our list.
    root_mean_squared_errors.append(root_mean_squared_error)
    # Finally, track the weights and biases over time.
    # Apply some math to ensure that the data and line are plotted neatly.
    y_extents = np.array([0, sample[my_label].max()])
    
    weight = linear_regressor.get_variable_value('linear/linear_model/%s/weights' % input_feature)[0]
    bias = linear_regressor.get_variable_value('linear/linear_model/bias_weights')

    x_extents = (y_extents - bias) / weight
    x_extents = np.maximum(np.minimum(x_extents,
                                      sample[my_feature].max()),
                           sample[my_feature].min())
    y_extents = weight * x_extents + bias
    plt.plot(x_extents, y_extents, color=colors[period]) 
  print("Model training finished.")

  # Output a graph of loss metrics over periods.
  plt.subplot(1, 2, 2)
  plt.ylabel('RMSE')
  plt.xlabel('Periods')
  plt.title("Root Mean Squared Error vs. Periods")
  plt.tight_layout()
  plt.plot(root_mean_squared_errors)

  # Output a table with calibration data.
  calibration_data = pd.DataFrame()
  calibration_data["predictions"] = pd.Series(predictions)
  calibration_data["targets"] = pd.Series(targets)
  display.display(calibration_data.describe())

  print("Final RMSE (on training data): %0.2f" % root_mean_squared_error)
#---------- 函数结束 ------------

#任务1：使 RMSE 不超过 180
train_model(
    learning_rate=0.00002,
    steps=500,
    batch_size=5
)

#任务2：尝试其他特征
train_model(
    learning_rate=0.00002,
    steps=1000,
    batch_size=5,
    input_feature="population"
)














