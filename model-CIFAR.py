import numpy as np  # to use numpy arrays
import tensorflow as tf  # to specify and run computation graphs
import tensorflow_datasets as tfds  # to load training data
import matplotlib.pyplot as plt  # to visualize data and draw plots
from tqdm import tqdm  # to track progress of loops
from tensorflow import keras


def define_model():

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

    flatten = tf.keras.layers.Flatten()
    dense1 = tf.keras.layers.Dense(256, activation="relu")
    dense2 = tf.keras.layers.Dense(128, activation="relu")
    output = tf.keras.layers.Dense(100, activation="softmax")
    model = tf.keras.Sequential(
        [
            hidden_1,
            pool_1,
            hidden_2,
            pool_2,
            hidden_3,
            pool_3,
            flatten,
            dense1,
            dense2,
            output,
        ]
    )
    return model

