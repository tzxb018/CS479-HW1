import numpy as np  # to use numpy arrays
import tensorflow as tf  # to specify and run computation graphs
import tensorflow_datasets as tfds  # to load training data
import matplotlib.pyplot as plt  # to visualize data and draw plots
from tqdm import tqdm  # to track progress of loops
from tensorflow import keras

model_cifar = __import__("model-CIFAR")
model_fmnist = __import__("model-FMNIST")
util_cifar = __import__("util-CIFAR")
util_fmnist = __import__("util-FMNIST")

dropping_rate = 0.1
actiavation_function = "relu"
model = model_cifar.define_model(dropping_rate, actiavation_function)
(td, tl), (vd, vl) = util_cifar.load_dataset()
model(td)
print(model.summary())
with open("report.txt", "w") as fh:
    # Pass the file handle in as a lambda function to make it callable
    model.summary(print_fn=lambda x: fh.write(x + "\n"))
