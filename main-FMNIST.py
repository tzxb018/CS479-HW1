# We'll start with our library imports...
from __future__ import print_function

import numpy as np  # to use numpy arrays
import tensorflow as tf  # to specify and run computation graphs
import tensorflow_datasets as tfds  # to load training data
import matplotlib.pyplot as plt  # to visualize data and draw plots
from tqdm import tqdm  # to track progress of loops
from tensorflow import keras
import os

print("hello world")
util_fmnist = __import__("util-FMNIST")
(train_images, train_labels), (test_images, test_labels) = util_fmnist.load_dataset()

model_fmnist = __import__("model-FMNIST")
model = model_fmnist.define_model()

# Directing the path for the checkpoint
checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path, save_weights_only=True, verbose=1
)

# Compiling the model
model.compile(
    optimizer="adam",
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

# Loads the weights
# model.load_weights(checkpoint_path)

model.fit(train_images, train_labels, epochs=1, callbacks=[cp_callback])
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)

print("\nTest accuracy:", test_acc)


# Evaluate the model
loss, acc = model.evaluate(test_images, test_labels, verbose=2)
