# # We'll start with our library imports...
# from __future__ import print_function

# import numpy as np  # to use numpy arrays
# import tensorflow as tf  # to specify and run computation graphs
# import tensorflow_datasets as tfds  # to load training data
# import matplotlib.pyplot as plt  # to visualize data and draw plots
# from tqdm import tqdm  # to track progress of loops
# from tensorflow import keras
# import os

# util_fmnist = __import__("util-FMNIST")
# (train_images, train_labels), (test_images, test_labels) = util_fmnist.load_dataset()

# model_fmnist = __import__("model-FMNIST")
# model = model_fmnist.define_model()

# # Directing the path for the checkpoint
# checkpoint_path = "training_1_FMNIST/cp.ckpt"
# checkpoint_dir = os.path.dirname(checkpoint_path)

# # Create a callback that saves the model's weights
# cp_callback = tf.keras.callbacks.ModelCheckpoint(
#     filepath=checkpoint_path, save_weights_only=True, verbose=1
# )

# # Compiling the model
# model.compile(
#     optimizer=tf.keras.optimizers.Adam(learning_rate=.001),
#     loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#     metrics=["accuracy"],
# )

# # Loads the weights
# # model.load_weights(checkpoint_path)

# history = model.fit(
#     train_images, train_labels, epochs=20, callbacks=[cp_callback], validation_split=0.1,
# )
# model.save("./fmnist_model_1")
# evaluation = model.evaluate(test_images, test_labels, verbose=2)

# print("\nTest accuracy:", evaluation[1])
# print(history)
# # Visualize history
# # Plot history: Loss
# plt.plot(history.history["accuracy"])
# plt.plot(history.history["val_accuracy"])
# plt.title("model accuracy")
# plt.ylabel("accuracy")
# plt.xlabel("epoch")
# plt.legend(["train", "test"], loc="upper left")
# plt.savefig("./fmnist_output/fmnist_accuracy.png")
# plt.clf()

# # Plot history: Accuracy
# plt.plot(history.history["loss"])
# plt.plot(history.history["val_loss"])
# plt.title("model loss")
# plt.ylabel("loss")
# plt.xlabel("epoch")
# plt.legend(["train", "test"], loc="upper left")
# plt.savefig("./fmnist_output/fmnist_lost.png")


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


drop_rate_arr = [0.1]
learning_rate_arr = [0.001, 0.0005, 0.0001]
history_arr = []
# drop_rate_arr = [0.25]
for dr in drop_rate_arr:
    for lr in learning_rate_arr:

        # ***********************************************************************************************
        learning_rate = lr
        activation_function = "relu"
        dropping_rate = dr
        epochs = 50

        model = model_fmnist.define_model_v2(dropping_rate, activation_function)

        learning_rate_str = str(learning_rate).replace(".", "x")
        dropping_rate_str = str(dropping_rate).replace(".", "x")

        # Directing the path for the checkpoint
        path = (
            "fmnist_model_lr_"
            + learning_rate_str
            + "dr_"
            + dropping_rate_str
            + "activation_"
            + str(activation_function)
        )
        checkpoint_dir = os.path.dirname("./models/training_" + path + "/cp.ckpt")

        # Create a callback that saves the model's weights
        cp_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_dir, save_weights_only=True, verbose=1
        )

        early_stopping = tf.keras.callbacks.EarlyStopping(monitor="loss", patience=5)

        # Compiling the original model model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"],
        )

        # Loads the weights
        # model.load_weights(checkpoint_path)

        history = model.fit(
            train_images,
            train_labels,
            epochs=epochs,
            callbacks=[cp_callback, early_stopping],
            validation_split=0.1,
        )
        model.save(path)
        evaluation = model.evaluate(test_images, test_labels, verbose=2)

        print("\nTest accuracy:", evaluation[1])
        history_arr.append(history)

        util_fmnist.evaluate_saved_model(
            path,
            test_images,
            test_labels,
            dropping_rate_str,
            dropping_rate,
            learning_rate,
            learning_rate_str,
        )

        # util_fmnist.graph_one(
        #     history, path, dropping_rate, dropping_rate_str, activation_function
        # )
    util_fmnist.graph_datasets(
        history_arr, path, dropping_rate, dropping_rate_str, activation_function
    )
    history_arr = []

