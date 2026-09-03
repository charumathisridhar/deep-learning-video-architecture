import tensorflow as tf
import time
import os
import json

from models import (
    build_shallow_cnn,
    build_deep_cnn,
    build_improved_cnn
)


DATASET_PATH = "frames"
IMAGE_SIZE = (128, 128)
BATCH_SIZE = 16
EPOCHS = 10
NUM_CLASSES = 3


# Load dataset
train_dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE
)

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE
)


AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(AUTOTUNE)
validation_dataset = validation_dataset.prefetch(AUTOTUNE)


def train_model(model, model_name):

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    print("\n==============================")
    print(model_name)
    print("==============================")

    start_time = time.time()

    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=EPOCHS
    )

    training_time = time.time() - start_time

    os.makedirs("results", exist_ok=True)

    model.save(f"results/{model_name}.keras")

    # Save training history
    with open(
        f"results/{model_name}_history.json",
        "w"
    ) as file:

        json.dump(
            history.history,
            file
        )

    print(
        f"{model_name} training time: "
        f"{training_time:.2f} seconds"
    )

    return training_time


if __name__ == "__main__":

    shallow_model = build_shallow_cnn()

    shallow_time = train_model(
        shallow_model,
        "shallow_cnn"
    )


    deep_model = build_deep_cnn()

    deep_time = train_model(
        deep_model,
        "deep_cnn"
    )


    improved_model = build_improved_cnn()

    improved_time = train_model(
        improved_model,
        "improved_cnn"
    )


    print("\n==============================")
    print("TRAINING TIME COMPARISON")
    print("==============================")

    print(f"Shallow CNN  : {shallow_time:.2f} seconds")
    print(f"Deep CNN     : {deep_time:.2f} seconds")
    print(f"Improved CNN : {improved_time:.2f} seconds")