import tensorflow as tf
import time
import os

from models import (
    build_shallow_cnn,
    build_deep_cnn,
    build_improved_cnn
)
# -----------------------------
# Settings
# -----------------------------

DATASET_PATH = "frames"

IMAGE_SIZE = (128, 128)
BATCH_SIZE = 16
EPOCHS = 10

NUM_CLASSES = 3


# -----------------------------
# Load dataset
# -----------------------------

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


# -----------------------------
# Improve performance
# -----------------------------

AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(AUTOTUNE)
validation_dataset = validation_dataset.prefetch(AUTOTUNE)


# -----------------------------
# Training function
# -----------------------------

def train_model(model, model_name):

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    print("\n==============================")
    print(model_name)
    print("==============================")

    model.summary()

    start_time = time.time()

    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=EPOCHS
    )

    end_time = time.time()

    training_time = end_time - start_time

    print(f"\nTraining time: {training_time:.2f} seconds")

    os.makedirs("results", exist_ok=True)

    model.save(f"results/{model_name}.keras")

    return history, training_time


# -----------------------------
# Train Shallow CNN
# -----------------------------

shallow_model = build_shallow_cnn(
    input_shape=(128, 128, 3),
    num_classes=NUM_CLASSES
)

shallow_history, shallow_time = train_model(
    shallow_model,
    "shallow_cnn"
)


# -----------------------------
# Train Deep CNN
# -----------------------------

deep_model = build_deep_cnn(
    input_shape=(128, 128, 3),
    num_classes=NUM_CLASSES
)

deep_history, deep_time = train_model(
    deep_model,
    "deep_cnn"
)

# -----------------------------
# Train Improved CNN
# -----------------------------

improved_model = build_improved_cnn(
    input_shape=(128, 128, 3),
    num_classes=NUM_CLASSES
)

improved_history, improved_time = train_model(
    improved_model,
    "improved_cnn"
)


# -----------------------------
# Final comparison
# -----------------------------

print("\n==============================")
print("MODEL COMPARISON")
print("==============================")

print(f"Shallow CNN training time: {shallow_time:.2f} seconds")
print(f"Deep CNN training time: {deep_time:.2f} seconds")
print(f"Improved CNN training time: {improved_time:.2f} seconds")