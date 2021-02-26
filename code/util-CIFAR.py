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

    return (train_images, train_labels), (test_images, test_labels)


def graph_datasets(
    history_arr, path, dropping_rate, dropping_rate_str, activation_function
):
    # Visualize history
    # https://www.machinecurve.com/index.php/2020/02/09/#how-to-build-a-convnet-for-cifar-10-and-cifar-100-classification-with-keras/
    # https://machinelearningmastery.com/display-deep-learning-model-training-history-in-keras/

    for history in history_arr:
        plt.plot(history.history["val_accuracy"])
    plt.title(
        "model accuracy (dropout rate="
        + str(dropping_rate)
        + ", activation="
        + str(activation_function)
        + ")"
    )
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(
        ["original", "learning_rate=.0005", "learning_rate=.0001"], loc="lower right",
    )
    plt.savefig("./cifar_output/accuracy_graph_" + dropping_rate_str + ".png")
    plt.clf()

    # Plot history: loss
    for history in history_arr:
        plt.plot(history.history["val_loss"])

    plt.title(
        "model accuracy (dropout rate="
        + str(dropping_rate)
        + ", activation="
        + str(activation_function)
        + ")"
    )
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(
        ["original", "learning_rate=.0005", "learning_rate=.0001"], loc="lower right",
    )
    plt.savefig("./cifar_output/loss_graph_" + dropping_rate_str + ".png")
    plt.clf()


def graph_one(
    history_original, path, dropping_rate, dropping_rate_str, activation_function
):
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


def evaluate_saved_model(
    path,
    test_images,
    test_labels,
    dropping_rate_str,
    dropping_rate,
    learning_rate,
    learnig_rate_str,
):
    model = keras.models.load_model("./cifar_models/" + path)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    evaluation = model.evaluate(test_images, test_labels, verbose=2)
    print("Retrieved model from " + path)
    print("\nTest accuracy: ", evaluation[1])

    f = open("cifar_output/cifar_accuarcies.txt", "a")
    f.write(
        "Dropping rate "
        + str(dropping_rate)
        + " learning rate "
        + str(learning_rate)
        + ": "
        + str(evaluation[1])
        + "\n"
    )
    f.close()

    y_pred = model.predict(test_images)
    y_pred = np.argmax(y_pred, axis=1)

    conf_mat = tf.math.confusion_matrix(test_labels, y_pred, num_classes=100)
    plt.title(
        "Confusion Matrix of Model (dropout rate="
        + str(dropping_rate)
        + ", activation="
        + "relu"
        + ")"
    )
    plt.imshow(conf_mat)
    plt.colorbar()
    plt.savefig(
        "cifar_output/confusion_matrix_"
        + dropping_rate_str
        + "_"
        + learnig_rate_str
        + ".png"
    )
    plt.clf()

    # plt.show()
