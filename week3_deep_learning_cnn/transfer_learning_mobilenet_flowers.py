# Transfer Learning using MobileNetV2 for Flower Classification
# Based on Week 3 resource: Transfer Learning with MobileNetV2.

import pathlib
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow import keras
from tensorflow.keras import layers

dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"

data_dir = keras.utils.get_file(
    "flower_photos",
    origin=dataset_url,
    cache_dir=".",
    untar=True
)

data_dir = pathlib.Path(data_dir)

batch_size = 32
img_height = 224
img_width = 224

train_ds = keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

val_ds = keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

class_names = train_ds.class_names
print("Classes:", class_names)

feature_extractor_url = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"

feature_extractor_layer = hub.KerasLayer(
    feature_extractor_url,
    input_shape=(224, 224, 3),
    trainable=False
)

model = keras.Sequential([
    layers.Rescaling(1.0 / 255),
    feature_extractor_layer,
    layers.Dense(len(class_names))
])

model.compile(
    optimizer="adam",
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"]
)

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=5
)

loss, accuracy = model.evaluate(val_ds)
print("Validation accuracy:", accuracy)