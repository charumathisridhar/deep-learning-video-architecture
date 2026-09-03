import tensorflow as tf
from tensorflow.keras import layers, models


def build_shallow_cnn(input_shape=(128, 128, 3), num_classes=3):

    model = models.Sequential([
        
        layers.Input(shape=input_shape),

        # Convolution Block 1
        layers.Conv2D(32, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),

        # Convolution Block 2
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),

        # Classification
        layers.Flatten(),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax")
    ])

    return model


if __name__ == "__main__":

    model = build_shallow_cnn()

    model.summary()