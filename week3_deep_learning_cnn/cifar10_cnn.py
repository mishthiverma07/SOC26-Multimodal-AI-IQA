# CIFAR-10 Image Classification using CNN
# Based on Week 3 resource: Image Classification using CNN on CIFAR-10.

import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np

(X_train, y_train), (X_test, y_test) = datasets.cifar10.load_data()

classes = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

print("Training data shape:", X_train.shape)
print("Test data shape:", X_test.shape)

y_train = y_train.reshape(-1)
y_test = y_test.reshape(-1)

def plot_sample(X, y, index):
    plt.figure(figsize=(3, 3))
    plt.imshow(X[index])
    plt.xlabel(classes[y[index]])
    plt.show()

plot_sample(X_train, y_train, 0)

# Normalize images
X_train = X_train / 255
X_test = X_test / 255

cnn = models.Sequential([
    layers.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(filters=64, kernel_size=(3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dense(64, activation="relu"),
    layers.Dense(10, activation="softmax")
])

cnn.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

cnn.fit(X_train, y_train, epochs=10)

loss, accuracy = cnn.evaluate(X_test, y_test)
print("Test accuracy:", accuracy)

y_predicted = cnn.predict(X_test)
y_classes = [np.argmax(element) for element in y_predicted]

print("Predicted class:", classes[y_classes[0]])
print("Actual class:", classes[y_test[0]])