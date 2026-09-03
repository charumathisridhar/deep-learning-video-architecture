import tensorflow as tf
import os

from models import build_shallow_cnn, build_deep_cnn


INPUT_SHAPE = (128, 128, 3)
NUM_CLASSES = 3


def analyze_model(model, model_name):

    print("\n" + "=" * 50)
    print(model_name)
    print("=" * 50)

    # Total parameters
    total_params = model.count_params()

    # Trainable parameters
    trainable_params = sum(
        tf.keras.backend.count_params(weight)
        for weight in model.trainable_weights
    )

    # Approximate memory for parameters
    memory_mb = (total_params * 4) / (1024 ** 2)

    print(f"Total Parameters     : {total_params:,}")
    print(f"Trainable Parameters : {trainable_params:,}")
    print(f"Approx. Parameter Memory : {memory_mb:.2f} MB")

    return total_params, memory_mb


# Build models

shallow_model = build_shallow_cnn(
    input_shape=INPUT_SHAPE,
    num_classes=NUM_CLASSES
)

deep_model = build_deep_cnn(
    input_shape=INPUT_SHAPE,
    num_classes=NUM_CLASSES
)


# Analyze

shallow_params, shallow_memory = analyze_model(
    shallow_model,
    "SHALLOW CNN"
)

deep_params, deep_memory = analyze_model(
    deep_model,
    "DEEP CNN"
)


# Comparison

print("\n" + "=" * 50)
print("ARCHITECTURAL COMPARISON")
print("=" * 50)

print(f"Shallow CNN Parameters : {shallow_params:,}")
print(f"Deep CNN Parameters    : {deep_params:,}")

print(f"\nShallow CNN Memory : {shallow_memory:.2f} MB")
print(f"Deep CNN Memory    : {deep_memory:.2f} MB")

print("\nConclusion:")

if deep_params > shallow_params:
    print("Deep CNN has more parameters and requires more memory.")

if deep_memory > shallow_memory:
    print("Deep CNN has higher parameter memory requirements.")