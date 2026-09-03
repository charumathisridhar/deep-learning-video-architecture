import json
import tensorflow as tf


MODELS = [
    "shallow_cnn",
    "deep_cnn",
    "improved_cnn"
]


def load_history(model_name):

    with open(
        f"results/{model_name}_history.json",
        "r"
    ) as file:

        return json.load(file)


print("\n" + "=" * 70)
print("FINAL MODEL PERFORMANCE")
print("=" * 70)

print(
    f"{'Model':<18}"
    f"{'Parameters':>15}"
    f"{'Train Acc':>15}"
    f"{'Val Acc':>15}"
)

print("-" * 70)


for model_name in MODELS:

    if model_name == "shallow_cnn":
        model = __import__(
            "models",
            fromlist=["build_shallow_cnn"]
        ).build_shallow_cnn()

    elif model_name == "deep_cnn":
        model = __import__(
            "models",
            fromlist=["build_deep_cnn"]
        ).build_deep_cnn()

    else:
        model = __import__(
            "models",
            fromlist=["build_improved_cnn"]
        ).build_improved_cnn()


    history = load_history(model_name)

    parameters = model.count_params()

    train_accuracy = history["accuracy"][-1]

    validation_accuracy = history["val_accuracy"][-1]


    print(
        f"{model_name:<18}"
        f"{parameters:>15,}"
        f"{train_accuracy * 100:>14.2f}%"
        f"{validation_accuracy * 100:>14.2f}%"
    )


print("=" * 70)