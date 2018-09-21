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
    output_targets['median_house_value_is_high'] = (
        california_housing_dataframe['median_house_value'] > 265000).astype(float)
    return output_targets
#training set
training_examples = preprocess_features(california_housing_dataframe.head(12000))
training_targets = preprocess_targets(california_housing_dataframe.head(12000))
validation_examples = preprocess_features(california_housing_dataframe.tail(5000))
validation_targets = preprocess_targets(california_housing_dataframe.tail(5000))
print('training examples')
display.display(training_examples.describe())
display.display(validation_examples.describe())
display.display(training_targets.describe())
display.display(validation_targets.describe())

def construct_features_columns(input_features):
    return set([tf.feature_column.numeric_column(my_feature)
                for my_feature in input_features])

def my_input_fn(features, targets, batch_size=1, shuffle=True, num_epochs=None):
    features = {key:np.array(value) for key,value in dict(features).items()}
    ds = Dataset.from_tensor_slices((features, targets))
    ds = ds.batch(batch_size).repeat(num_epochs)
    if shuffle:
        ds = ds.shuffle(10000)
    features, labels = ds.make_one_shot_iterator().get_next()
    return features, labels

def model_size(estimator):
    variables = estimator.get_variable_names()
    size=0
    for variable in variables:
        if not any(x in variable
                    for x in ['global_step',
                              'centered_bias_weight',
                              'bias_weight',
                              'Ftrl']):
            size+=np.count_nonzero(estimator.get_variable_value(variable))
        return size



def train_linear_regressor_model(
    learning_rate,
    steps,
    batch_size,
    training_examples,
    training_targets,
    validation_examples,
    validation_targets):
    periods = 10
    steps_per_period = steps/periods
    my_optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
    linear_regressor = tf.estimator.LinearRegressor(
        feature_columns = construct_features_columns(training_examples),
        optimizer = my_optimizer)

    training_input_fn = lambda: my_input_fn(training_examples,
                                            training_targets['median_house_value_is_high'],
                                            batch_size=batch_size)
    predict_training_input_fn = lambda: my_input_fn(training_examples,
                                                    training_targets['median_house_value_is_high'],
                                                    num_epochs=1,
                                                    shuffle=False)
    predict_validation_input_fn = lambda: my_input_fn(validation_examples,
                                                    validation_targets['median_house_value_is_high'],
                                                    num_epochs=1,
                                                    shuffle=False)

    print('Training model ------------------------------------')
    training_rmse = []
    validation_rmse = []
    for period in range(0, periods):
        linear_regressor.train(input_fn=training_input_fn,
                                steps=steps_per_period)
        training_predictions = linear_regressor.predict(input_fn = predict_training_input_fn)
        training_predictions = np.array([item['predictions'][0] for item in training_predictions])

        validation_predictions = linear_regressor.predict(input_fn = predict_validation_input_fn)
        validation_predictions = np.array([item['predictions'][0] for item in validation_predictions])

        training_rmse_def = math.sqrt(
            metrics.mean_squared_error(training_predictions, training_targets))
        validation_rmse_def = math.sqrt(
            metrics.mean_squared_error(validation_predictions, validation_targets))
        training_rmse.append(training_rmse_def)
        validation_rmse.append(validation_rmse_def)
    print('Training Finished -------------------------------')
    plt.ylabel('rmse')
    plt.xlabel('periods')
    plt.title('rmse x periods')
    plt.tight_layout()
    plt.plot(training_rmse, label='training')
    plt.plot(validation_rmse, label='validation')
    plt.legend()
    plt.show()
    return linear_regressor

def train_linear_classifier_model(
    learning_rate,
    regularization_strength,
    steps,
    batch_size,
    training_examples,
    training_targets,
    validation_examples,
    validation_targets):
    periods = 10
    steps_per_period = steps/ periods

    my_optimizer = tf.train.FtrlOptimizer(learning_rate=learning_rate, l1_regularization_strength=regularization_strength)
    my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
    linear_classifier = tf.estimator.LinearClassifier(
        feature_columns=construct_features_columns(training_examples),
        optimizer=my_optimizer)
    training_input_fn = lambda: my_input_fn(training_examples,
        training_targets['median_house_value_is_high'],
        batch_size=batch_size)
    predict_training_input_fn = lambda: my_input_fn(training_examples,
        training_targets['median_house_value_is_high'],
        num_epochs=1,
        shuffle=False)
    predict_validation_input_fn = lambda: my_input_fn(validation_examples,
        validation_targets['median_house_value_is_high'],
        num_epochs=1,
        shuffle=False)
    print('training------------')
    training_log_losses = []
    validation_log_losses = []
    for period in range(0, periods):
        linear_classifier.train(
            input_fn=training_input_fn,
            steps=steps_per_period)

        training_probabilities= linear_classifier.predict(input_fn=predict_training_input_fn)
        training_probabilities = np.array([item['probabilities'] for item in training_probabilities])
        validation_probabilities = linear_classifier.predict(input_fn=predict_validation_input_fn)
        validation_probabilities = np.array([item['probabilities'] for item in validation_probabilities])

        training_log_loss = metrics.log_loss(training_targets, training_probabilities)
        validation_log_loss = metrics.log_loss(validation_targets, validation_probabilities)
        training_log_losses.append(training_log_loss)
        validation_log_losses.append(validation_log_loss)

    evaluation_metrics = linear_classifier.evaluate(input_fn=predict_validation_input_fn)
    print('auc validation %0.2f' %evaluation_metrics['auc'])
    print('accuracy validation %0.2f' %evaluation_metrics['accuracy'])

    plt.ylabel('LogLoss')
    plt.xlabel('periods')
    plt.title('LogLoss x periods')
    plt.tight_layout()
    plt.plot(training_log_losses, label='training')
    plt.plot(validation_log_losses, label='validation')
    plt.legend()
    plt.show()
    return linear_classifier

linear_classifier = train_linear_classifier_model(
        learning_rate=0.00005,
        regularization_strength=0.1,
        steps=300,
        batch_size=100,
        training_examples=training_examples,
        training_targets=training_targets,
        validation_examples=validation_examples,
        validation_targets=validation_targets)
print('model size', model_size(linear_classifier))
