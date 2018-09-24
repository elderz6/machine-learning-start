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

# plt.figure(figsize=(13,8))
# ax = plt.subplot(1, 2, 1)
# ax.set_title('validation data')
#
# ax.set_autoscaley_on(False)
# ax.set_ylim([32, 43])
# ax.set_autoscalex_on(False)
# ax.set_xlim([-126, -112])
# plt.scatter(training_examples['longitude'],
#     training_examples['latitude'],
#     cmap='coolwarm',
#     c=training_targets['median_house_value'] / training_targets['median_house_value'].max())
# _ = plt.plot()
# #plt.show()
#linear regression training function
def my_input_fn(features, targets, batch_size=1, shuffle=True, num_epochs=None):
    #convert pandas data to np array
    features = {key:np.array(value) for key,value in dict(features).items()}
    ds = Dataset.from_tensor_slices((features,targets))
    ds = ds.batch(batch_size).repeat(num_epochs)
    #shuffle if specified
    if shuffle:
        ds = ds.shuffle(1000)
    features, labels = ds.make_one_shot_iterator().get_next()
    return features, labels

def construct_features_columns():
    households = tf.feature_column.numeric_column('households')
    longitude = tf.feature_column.numeric_column('longitude')
    latitude = tf.feature_column.numeric_column('latitude')
    housing_median_age = tf.feature_column.numeric_column('housing_median_age')
    median_income = tf.feature_column.numeric_column('median_income')
    rooms_per_person = tf.feature_column.numeric_column('rooms_per_person')

    bucketized_households = tf.feature_column.bucketized_column(households,
        boundaries=get_quantile_based_boundaries(california_housing_dataframe['households'], 7))
    bucketized_longitude = tf.feature_column.bucketized_column(longitude,
        boundaries=get_quantile_based_boundaries(california_housing_dataframe['longitude'], 10))

    feature_columns = set([bucketized_longitude, bucketized_households])
    return feature_columns
    # return set([tf.feature_column.numeric_column(my_feature)
    #             for my_feature in input_features])

def train_model(
    learning_rate,
    steps,
    batch_size,
    hidden_units,
    feature_columns,
    training_examples,
    training_targets,
    validation_examples,
    validation_targets):
    periods = 20
    steps_per_period = steps/periods
    #creating linear regressor object and optimizer
    my_optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
    dnn_regressor = tf.estimator.DNNRegressor(
        feature_columns = construct_features_columns(),
        hidden_units=hidden_units,
        optimizer = my_optimizer)
    #input functions
    training_input_fn = lambda: my_input_fn(
        training_examples,
        training_targets['median_house_value'],
        batch_size=batch_size)
    predict_training_input_fn = lambda: my_input_fn(
        training_examples,
        training_targets['median_house_value'],
        num_epochs=1,
        shuffle=False)
    predict_validation_input_fn = lambda: my_input_fn(
        validation_examples,
        validation_targets['median_house_value'],
        num_epochs=1,
        shuffle=False)
    #train the model and check loss
    print('training model')
    training_rmse = []
    validation_rmse = []
    for period in range(0, periods):
        dnn_regressor.train(
            input_fn=training_input_fn,
            steps=steps_per_period)
        training_predictions = dnn_regressor.predict(input_fn=predict_training_input_fn)
        training_predictions = np.array([item['predictions'][0] for item in training_predictions])
        validation_predictions = dnn_regressor.predict(input_fn=predict_validation_input_fn)
        validation_predictions = np.array([item['predictions'][0] for item in validation_predictions])

        training_rmse_def = math.sqrt(
            metrics.mean_squared_error(training_predictions, training_targets))
        validation_rmse_def = math.sqrt(
            metrics.mean_squared_error(validation_predictions, validation_targets))

        print('period %02d : %0.2f' % (period, training_rmse_def))

        training_rmse.append(training_rmse_def)
        validation_rmse.append(validation_rmse_def)

    print('training finished', training_rmse, validation_rmse)

    plt.ylabel("RMSE")
    plt.xlabel("Periods")
    plt.title("Root Mean Squared Error vs. Periods")
    plt.tight_layout()
    plt.plot(training_rmse, label="training")
    plt.plot(validation_rmse, label="validation")
    plt.legend()
    plt.show()
    return dnn_regressor

def get_quantile_based_boundaries(feature_values, num_buckets):
    boundaries = np.arange(1.0, num_buckets) / num_buckets
    quantiles = feature_values.quantile(boundaries)
    return [quantiles[q] for q in quantiles.keys()]

def linear_scale(series):
    min_val = series.min()
    max_val = series.max()
    scale = (max_val - min_val) / 2.0
    return series.apply(lambda x:((x - min_val)/ scale) - 1.0)

def normalize_linear_scale(examples_dataframe):
    processed_features = pd.DataFrame()
    processed_features["latitude"] = linear_scale(examples_dataframe["latitude"])
    processed_features["longitude"] = linear_scale(examples_dataframe["longitude"])
    processed_features["housing_median_age"] = linear_scale(examples_dataframe["housing_median_age"])
    processed_features["total_rooms"] = linear_scale(examples_dataframe["total_rooms"])
    processed_features["total_bedrooms"] = linear_scale(examples_dataframe["total_bedrooms"])
    processed_features["population"] = linear_scale(examples_dataframe["population"])
    processed_features["households"] = linear_scale(examples_dataframe["households"])
    processed_features["median_income"] = linear_scale(examples_dataframe["median_income"])
    processed_features["rooms_per_person"] = linear_scale(examples_dataframe["rooms_per_person"])
    return processed_features
normalized_dataframe = normalize_linear_scale(preprocess_features(california_housing_dataframe))
normalized_training_examples = normalized_dataframe.head(12000)
normalized_validation_examples = normalized_dataframe.tail(5000)

linear_regressor = train_model(
    learning_rate=0.01,
    steps=500,
    hidden_units=[10, 2],
    batch_size=1,
    feature_columns=construct_features_columns(),
    training_examples=normalized_training_examples,
    training_targets=training_targets,
    validation_examples=normalized_validation_examples,
    validation_targets=validation_targets)
