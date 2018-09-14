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
#define dataframe
california_housing_dataframe = pd.read_csv("https://download.mlcc.google.com/mledu-datasets/california_housing_train.csv", sep=",")

california_housing_dataframe = california_housing_dataframe.reindex(
    np.random.permutation(california_housing_dataframe.index))
california_housing_dataframe['median_house_value'] /= 1000.0
california_housing_dataframe
#print the thing
california_housing_dataframe.describe()
print(california_housing_dataframe)
#start and configure the feature
my_feature = california_housing_dataframe[['total_rooms']]
feature_columns = [tf.feature_column.numeric_column("total_rooms")]
#define label
targets = california_housing_dataframe['median_house_value']
#set gradient descent, linear regressor
my_optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.000001)
my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)

linear_regressor = tf.estimator.LinearRegressor(
    feature_columns = feature_columns,
    optimizer = my_optimizer
)
#input function
def my_input_fn(features, targets, batch_size=1, shuffle=True, num_epochs=None):
    features = {key:np.array(value) for key, value in dict(features).items()}
    ds = Dataset.from_tensor_slices((features, targets))
    ds = ds.batch(batch_size).repeat(num_epochs)

    if shuffle:
        ds = ds.shuffle(buffer_size=10000)

    features, labels = ds.make_one_shot_iterator().get_next()
    return features, labels
#train the model
_ = linear_regressor.train(
    input_fn = lambda:my_input_fn(my_feature, targets),
    steps=100
)
#input function for predictions
prediction_input_fn = lambda: my_input_fn(my_feature, targets, num_epochs=1, shuffle=False)
#call precit
predictions = linear_regressor.predict(input_fn=prediction_input_fn)
#format predictions to numpy array
predictions = np.array([item['predictions'][0] for item in predictions])
#mean squared error
mean_squared_error = metrics.mean_squared_error(predictions, targets)
root_mean_squared_error = math.sqrt(mean_squared_error)
print('MSE: %.3f' % mean_squared_error)
#compare rmse to median value
min_house_value = california_housing_dataframe['median_house_value'].min()
max_house_value = california_housing_dataframe['median_house_value'].max()
min_max_difference = max_house_value - min_house_value

print('min value %.3f'%min_house_value)
print('max value %.3f'%max_house_value)
print('min max dif value %.3f'%min_max_difference)
print('ROOT MSE %.3f'%root_mean_squared_error)
#calibration to reduce model error
calibration_data = pd.DataFrame()
california_housing_dataframe['predictions'] = pd.Series(predictions)
calibration_data['targets'] = pd.Series(targets)

sample = california_housing_dataframe.sample(n=300)
print(sample)

x_0 = sample['total_rooms'].min()
x_1 = sample['total_rooms'].max()

weight = linear_regressor.get_variable_value('linear/linear_model/total_rooms/weights')[0]
bias = linear_regressor.get_variable_value('linear/linear_model/bias_weights')

y_0 = weight * x_0 + bias
y_1 = weight * x_1 + bias

plt.plot([x_0, x_1], [y_0, y_1], c='r')

plt.ylabel('median_house_value')
plt.xlabel('total rooms')

plt.scatter(sample['total_rooms'], sample['median_house_value'])

plt.show()
