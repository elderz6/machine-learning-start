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
tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format
#define dataframe
california_housing_dataframe = pd.read_csv("https://download.mlcc.google.com/mledu-datasets/california_housing_train.csv", sep=",")
california_housing_dataframe = california_housing_dataframe.reindex(
    np.random.permutation(california_housing_dataframe.index))
#return dataframe with features to be used for the model
def preprocess_features(california_housing_dataframe):
    selected_features = california_housing_dataframe[
    ['latitude',
     'longitude',
     'housing_median_age',
     'total_rooms',
     'total_bedrooms',
     'population',
     'households',
     'median_income'
     ]]
    processed_features = selected_features.copy()
     #creating new "synthetic" feature
    processed_features['rooms_per_person'] = (
        california_housing_dataframe['total_rooms']/
        california_housing_dataframe['population'])
    return processed_features
#prepares target features
def preprocess_targets(california_housing_dataframe):
    output_targets = pd.DataFrame()
    output_targets['median_house_value'] = (
        california_housing_dataframe['median_house_value'] / 1000.0)
    return output_targets
#training set
training_examples = preprocess_features(california_housing_dataframe.head(12000))
print(training_examples)
training_targets = preprocess_targets(california_housing_dataframe.head(12000))
print(training_targets)
validation_examples = preprocess_features(california_housing_dataframe.tail(5000))
print(validation_examples)
validation_targets = preprocess_targets(california_housing_dataframe.tail(5000))
print(validation_targets)

plt.figure(figsize=(13,8))
ax = plt.subplot(1, 2, 1)
ax.set_title('validation data')

ax.set_autoscaley_on(False)
ax.set_ylim([32, 43])
ax.set_autoscalex_on(False)
ax.set_xlim([-126, -112])
plt.scatter(training_examples['longitude'],
    training_examples['latitude'],
    cmap='coolwarm',
    c=training_targets['median_house_value'] / training_targets['median_house_value'].max())

_ = plt.plot()
plt.show()
