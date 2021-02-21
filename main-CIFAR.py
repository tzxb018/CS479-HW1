# We'll start with our library imports...
from __future__ import print_function

import numpy as np  # to use numpy arrays
import tensorflow as tf  # to specify and run computation graphs
import tensorflow_datasets as tfds  # to load training data
import matplotlib.pyplot as plt  # to visualize data and draw plots
from tqdm import tqdm  # to track progress of loops
from tensorflow import keras
import os

util_fmnist = __import__("util-CIFAR")
(train_images, train_labels), (test_images, test_labels) = util_fmnist.load_dataset()

model_fmnist = __import__("model-CIFAR")

learning_rate = 0.001
activation_function = "relu"
dropping_rate = 0.1
# ***********************************************************************************************
model = model_fmnist.define_model(dropping_rate, activation_function)

learning_rate_str = str(learning_rate).replace(".", "x")
dropping_rate_str = str(dropping_rate).replace(".", "x")

# Directing the path for the checkpoint
path = (
    "cifar_model_lr_"
    + learning_rate_str
    + "dr_"
    + dropping_rate_str
    + "activation_"
    + str(activation_function)
)
checkpoint_dir = os.path.dirname("training_" + path + "/cp.ckpt")

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=path, save_weights_only=True, verbose=1
)

# Compiling the original model model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

# Loads the weights
# model.load_weights(checkpoint_path)

history_original = model.fit(
    train_images,
    train_labels,
    epochs=25,
    callbacks=[cp_callback],
    validation_split=0.1,
)
model.save(path)
evaluation_original = model.evaluate(test_images, test_labels, verbose=2)

print("\nTest accuracy:", evaluation_original[1])

# ***********************************************************************************************
# Directing the path for the checkpoint
# checkpoint_path = "training_cifar_model_v2_0005_lr_2_dr_relu_act/cp.ckpt"
# checkpoint_dir = os.path.dirname(checkpoint_path)

# # Create a callback that saves the model's weights
# cp_callback = tf.keras.callbacks.ModelCheckpoint(
#     filepath=checkpoint_path, save_weights_only=True, verbose=1
# )

# # Compiling the model for different learning rate
# model = model_fmnist.define_model_v2(0.2, "relu")
# model.compile(
#     optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
#     loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#     metrics=["accuracy"],
# )

# # Loads the weights
# # model.load_weights(checkpoint_path)

# history_0005 = model.fit(
#     train_images,
#     train_labels,
#     epochs=30,
#     callbacks=[cp_callback],
#     validation_split=0.1,
# )
# model.save("./cifar_model_v2_0005_lr_2_dr_relu_act")
# evaluation_0005 = model.evaluate(test_images, test_labels, verbose=2)

# print("\nTest accuracy:", evaluation_0005[1])

# ***********************************************************************************************
# Directing the path for the checkpoint
# checkpoint_path = "training_cifar_model_v2_0001_lr_2_dr_relu_act/cp.ckpt"
# checkpoint_dir = os.path.dirname(checkpoint_path)

# # Create a callback that saves the model's weights
# cp_callback = tf.keras.callbacks.ModelCheckpoint(
#     filepath=checkpoint_path, save_weights_only=True, verbose=1
# )

# # Compiling the model for different learning rate
# model = model_fmnist.define_model_v2(0.2, "relu")
# model.compile(
#     optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
#     loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#     metrics=["accuracy"],
# )

# # Loads the weights
# # model.load_weights(checkpoint_path)

# history_01 = model.fit(
#     train_images,
#     train_labels,
#     epochs=30,
#     callbacks=[cp_callback],
#     validation_split=0.1,
# )
# model.save("./cifar_model_v2_0001_lr_2_dr_relu_act")
# evaluation_01 = model.evaluate(test_images, test_labels, verbose=2)

# print("\nTest accuracy:", evaluation_01[1])

# ***********************************************************************************************
# Visualize history
# https://www.machinecurve.com/index.php/2020/02/09/how-to-build-a-convnet-for-cifar-10-and-cifar-100-classification-with-keras/
# https://machinelearningmastery.com/display-deep-learning-model-training-history-in-keras/
# Visualize history
# Plot history: Loss
# plt.plot(history_original.history["val_accuracy"])
# plt.plot(history_0005.history["val_accuracy"])
# plt.plot(history_01.history["val_accuracy"])
# plt.title("model accuracy v2 (dropout rate=.2, activation='relu')")
# plt.ylabel("accuracy")
# plt.xlabel("epoch")
# plt.legend(
#     ["original", "learning_rate=.0005", "learning_rate=.0001",], loc="lower right",
# )
# plt.savefig("./cifar_output/cifar_accuracy_v2_2_dr_relu_act.png")
# plt.clf()

# # Plot history: loss
# plt.plot(history_original.history["val_loss"])
# plt.plot(history_0005.history["val_loss"])
# plt.plot(history_01.history["val_loss"])
# plt.title("model loss v2 (dropout rate=.2, activation='relu')")
# plt.ylabel("loss")
# plt.xlabel("epoch")
# plt.legend(
#     ["original", "learning_rate=.0005", "learning_rate=.0001",], loc="lower right",
# )
# plt.savefig("./cifar_output/cifar_loss_v2_2_dr_relu_act.png")


plt.plot(history_original.history["val_accuracy"])
plt.plot(history_original.history["accuracy"])
plt.title("model accuracy v2 (dropout rate=.1, activation='relu')")
plt.ylabel("accuracy")
plt.xlabel("epoch")
plt.legend(
    ["validation accuracy, accuracy",], loc="lower right",
)
plt.savefig("./cifar_output/" + path + "_accuracy_graph.png")
plt.clf()

# Plot history: loss
plt.plot(history_original.history["val_loss"])
plt.plot(history_original.history["loss"])
plt.title("model loss v2 (dropout rate=.1, activation='relu')")
plt.ylabel("loss")
plt.xlabel("epoch")
plt.legend(
    ["validation loss, loss",], loc="lower right",
)
plt.savefig("./cifar_output/" + path + "_loss_graph.png")
plt.clf()
