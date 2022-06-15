import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import (Conv2D, Dense, Dropout, Flatten,
                                     MaxPooling2D)
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

checkpoint_filepath = "models/yixian_six_class/9_6_2pm_yixian6class"
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=False,
    monitor="val_accuracy",
    mode="max",
    save_best_only=True,
)

emotion_model = Sequential()

emotion_model.add(
    Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=(90, 90, 3))
)
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))

emotion_model.add(Conv2D(64, kernel_size=(3, 3), activation="relu"))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.2))

emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation="relu"))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))

emotion_model.add(Dropout(0.2))
emotion_model.add(Flatten())

emotion_model.add(Dense(1024, activation="relu"))
emotion_model.add(Dropout(0.5))
emotion_model.add(Dense(6, activation="softmax"))


cv2.ocl.setUseOpenCL(False)

emotion_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

history = emotion_model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=2,
    validation_steps=5,
    verbose=1,
    callbacks=[model_checkpoint_callback],
)


emotion_model.save(constants.MODEL_CHECKPOINTS)

acc = history.history["accuracy"]
val_acc = history.history["val_accuracy"]

loss = history.history["loss"]
val_loss = history.history["val_loss"]
