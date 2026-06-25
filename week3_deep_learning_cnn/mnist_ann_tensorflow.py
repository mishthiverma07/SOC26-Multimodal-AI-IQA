# MNIST Handwritten Digit Recognition using TensorFlow/Keras
# Based on Week 3 resource: Handwritten Digit Recognition using ANN.

import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np

(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

print("Training data shape:", X_train.shape)
print("Test data shape:", X_test.shape)

plt.matshow(X_train[0])
plt.title(f"Actual label: {y_train[0]}")
plt.show()

# Scaling pixel values from 0-255 to 0-1
X_train = X_train / 255
X_test = X_test / 255

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(100, activation="relu"),
    keras.layers.Dense(10, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(X_train, y_train, epochs=5)

loss, accuracy = model.evaluate(X_test, y_test)
print("Test accuracy:", accuracy)

y_predicted = model.predict(X_test)

print("Predicted digit:", np.argmax(y_predicted[0]))
print("Actual digit:", y_test[0])