import numpy as np  # to use numpy arrays
import tensorflow as tf  # to specify and run computation graphs
import tensorflow_datasets as tfds  # to load training data
import matplotlib.pyplot as plt  # to visualize data and draw plots
from tqdm import tqdm  # to track progress of loops
from tensorflow import keras


def define_model(dropout_rate, activation_function):

    hidden_1 = tf.keras.layers.Conv2D(
        filters=32,
        kernel_size=3,
        padding="same",
        activation=tf.nn.relu,
        name="hidden_1",
    )
    pool_1 = tf.keras.layers.MaxPool2D(padding="same", pool_size=(2, 2))

    hidden_2 = tf.keras.layers.Conv2D(
        filters=64,
        kernel_size=3,
        padding="same",
        activation=tf.nn.relu,
        name="hidden_2",
    )
    pool_2 = tf.keras.layers.MaxPool2D(padding="same", pool_size=(2, 2))

    hidden_3 = tf.keras.layers.Conv2D(
        filters=128,
        kernel_size=3,
        padding="same",
        activation=tf.nn.relu,
        name="hidden_3",
    )
    pool_3 = tf.keras.layers.MaxPool2D(padding="same", pool_size=(2, 2))
    dropout1 = tf.keras.layers.Dropout(rate=dropout_rate)
    flatten = tf.keras.layers.Flatten()
    dense1 = tf.keras.layers.Dense(256, activation=activation_function)
    dense2 = tf.keras.layers.Dense(128, activation=activation_function)
    output = tf.keras.layers.Dense(100)
    model = tf.keras.Sequential(
        [
            hidden_1,
            pool_1,
            hidden_2,
            pool_2,
            hidden_3,
            pool_3,
            dropout1,
            flatten,
            dense1,
            dense2,
            output,
        ]
    )
    return model


def define_model_v2(dropout_rate, activation_function):
    conv_classifier = tf.keras.Sequential()
    hidden_1 = tf.keras.layers.Conv2D(
        filters=32,
        kernel_size=3,
        padding="same",
        activation=tf.nn.relu,
        name="hidden_1",
    )
    hidden_2 = tf.keras.layers.Conv2D(
        filters=64,
        kernel_size=3,
        padding="same",
        activation=tf.nn.relu,
        name="hidden_2",
    )
    pool_1 = tf.keras.layers.MaxPool2D(padding="same")
    hidden_3 = tf.keras.layers.Conv2D(
        filters=128,
        kernel_size=3,
        padding="same",
        activation=tf.nn.relu,
        name="hidden_3",
    )
    hidden_4 = tf.keras.layers.Conv2D(
        filters=256,
        kernel_size=3,
        padding="same",
        activation=tf.nn.relu,
        name="hidden_4",
    )
    pool_2 = tf.keras.layers.MaxPool2D(padding="same")
    flatten = tf.keras.layers.Flatten()
    dense1 = tf.keras.layers.Dense(256, activation=activation_function)
    dense2 = tf.keras.layers.Dense(128, activation=activation_function)
    output = tf.keras.layers.Dense(10)
    dropout1 = tf.keras.layers.Dropout(rate=dropout_rate)
    dropout2 = tf.keras.layers.Dropout(rate=dropout_rate)
    conv_classifier = tf.keras.Sequential(
        [
            hidden_1,
            hidden_2,
            pool_1,
            hidden_3,
            hidden_4,
            pool_2,
            dense1,
            dense2,
            flatten,
            output,
        ]
    )
    return conv_classifier
