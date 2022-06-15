import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from .. import constants

physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)

train_data_gen = ImageDataGenerator(rescale=1.0 / 255)
validation_data_gen = ImageDataGenerator(rescale=1.0 / 255)

train_generator = train_data_gen.flow_from_directory(
    f"{constants.TRAIN_DATA_PATH}",
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical",
    shuffle=True,
)

validation_generator = validation_data_gen.flow_from_directory(
    f"{constants.VAL_DATA_PATH}",
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical",
)

checkpoint_filepath = constants.MODEL_CHECKPOINTS

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=False,
    monitor="val_loss",
    mode="min",
    save_best_only=True,
)


base_model = tf.keras.applications.vgg16.VGG16(
    input_shape=(224, 224, 3), include_top=False, weights="imagenet"
)

base_model.trainable = False

print(base_model.summary())

emotion_model = keras.Sequential(
    [
        base_model,
        layers.Flatten(),
        layers.Dense(1024, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(512, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(6, activation="softmax"),
    ]
)

print(emotion_model.summary())

emotion_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

history = emotion_model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=50,
    validation_steps=5,
    verbose=1,
)

emotion_model.save(constants.MODEL_CHECKPOINTS)

acc = history.history["accuracy"]
val_acc = history.history["val_accuracy"]

loss = history.history["loss"]
val_loss = history.history["val_loss"]
