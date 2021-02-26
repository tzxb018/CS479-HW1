import numpy as np  # to use numpy arrays
import tensorflow as tf  # to specify and run computation graphs
import tensorflow_datasets as tfds  # to load training data
import matplotlib.pyplot as plt  # to visualize data and draw plots
from tqdm import tqdm  # to track progress of loops
from tensorflow import keras


def define_model(dropping_rate, activation_function):

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(10),
        ]
    )
    return model


def define_model_v2(dropping_rate, activation_function):
    scale = 1.0 / 255
    preprocess = tf.keras.layers.experimental.preprocessing.Rescaling(scale=scale)
    model = tf.keras.Sequential()
    model = tf.keras.Sequential(
        [
            preprocess,
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(128, activation=activation_function),
            tf.keras.layers.Dense(256, activation=activation_function),
            tf.keras.layers.Dropout(dropping_rate),
            tf.keras.layers.Dense(512, activation=activation_function),
            tf.keras.layers.Dropout(dropping_rate),
            tf.keras.layers.Dense(1024, activation=activation_function),
            tf.keras.layers.Dropout(dropping_rate),
            tf.keras.layers.Dense(10),
        ]
    )
    return model
