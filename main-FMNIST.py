# We'll start with our library imports...
from __future__ import print_function

import numpy as np  # to use numpy arrays
import tensorflow as tf  # to specify and run computation graphs
import tensorflow_datasets as tfds  # to load training data
import matplotlib.pyplot as plt  # to visualize data and draw plots
from tqdm import tqdm  # to track progress of loops
from tensorflow import keras
import os

util_fmnist = __import__("util-FMNIST")
(train_images, train_labels), (test_images, test_labels) = util_fmnist.load_dataset()

model_fmnist = __import__("model-FMNIST")
model = model_fmnist.define_model()

# Directing the path for the checkpoint
checkpoint_path = "training_1_FMNIST/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path, save_weights_only=True, verbose=1
)

# Compiling the model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

# Loads the weights
# model.load_weights(checkpoint_path)

history = model.fit(
    train_images, train_labels, epochs=2, callbacks=[cp_callback], validation_split=0.1,
)
model.save("./fmnist_model_1")
evaluation = model.evaluate(test_images, test_labels, verbose=2)

print("\nTest accuracy:", evaluation[1])
print(history)
# Visualize history
# Plot history: Loss
plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])
plt.title("model accuracy")
plt.ylabel("accuracy")
plt.xlabel("epoch")
plt.legend(["train", "test"], loc="upper left")
plt.savefig("./fmnist_output/fmnist_accuracy.png")
plt.clf()

# Plot history: Accuracy
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("model loss")
plt.ylabel("loss")
plt.xlabel("epoch")
plt.legend(["train", "test"], loc="upper left")
plt.savefig("./fmnist_output/fmnist_lost.png")
