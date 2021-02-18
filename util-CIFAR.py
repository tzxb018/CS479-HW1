import numpy as np  # to use numpy arrays
import tensorflow as tf  # to specify and run computation graphs
import tensorflow_datasets as tfds  # to load training data
import matplotlib.pyplot as plt  # to visualize data and draw plots
from tqdm import tqdm  # to track progress of loops
from tensorflow import keras


def load_dataset():

    cifar100 = tf.keras.datasets.cifar100

    (train_images, train_labels), (test_images, test_labels) = cifar100.load_data()

    # plt.figure()
    # plt.imshow(train_images[0])
    # plt.colorbar()
    # plt.grid(False)
    # plt.show()

    # preprocessing the data
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    return (train_images, train_labels), (test_images, test_labels)

