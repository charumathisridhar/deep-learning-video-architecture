import json
import matplotlib.pyplot as plt
import os


os.makedirs("results", exist_ok=True)


def load_history(model_name):

    with open(
        f"results/{model_name}_history.json",
        "r"
    ) as file:

        return json.load(file)


shallow = load_history("shallow_cnn")
deep = load_history("deep_cnn")
improved = load_history("improved_cnn")


# -----------------------------
# Accuracy
# -----------------------------

plt.figure(figsize=(8, 5))

plt.plot(
    shallow["accuracy"],
    label="Shallow CNN"
)

plt.plot(
    deep["accuracy"],
    label="Deep CNN"
)

plt.plot(
    improved["accuracy"],
    label="Improved CNN"
)

plt.xlabel("Epoch")
plt.ylabel("Training Accuracy")
plt.title("Training Accuracy Comparison")

plt.legend()
plt.grid()

plt.savefig(
    "results/final_accuracy_comparison.png"
)

plt.show()


# -----------------------------
# Validation Accuracy
# -----------------------------

plt.figure(figsize=(8, 5))

plt.plot(
    shallow["val_accuracy"],
    label="Shallow CNN"
)

plt.plot(
    deep["val_accuracy"],
    label="Deep CNN"
)

plt.plot(
    improved["val_accuracy"],
    label="Improved CNN"
)

plt.xlabel("Epoch")
plt.ylabel("Validation Accuracy")
plt.title("Validation Accuracy Comparison")

plt.legend()
plt.grid()

plt.savefig(
    "results/final_validation_accuracy.png"
)

plt.show()


# -----------------------------
# Loss
# -----------------------------

plt.figure(figsize=(8, 5))

plt.plot(
    shallow["val_loss"],
    label="Shallow CNN"
)

plt.plot(
    deep["val_loss"],
    label="Deep CNN"
)

plt.plot(
    improved["val_loss"],
    label="Improved CNN"
)

plt.xlabel("Epoch")
plt.ylabel("Validation Loss")
plt.title("Validation Loss Comparison")

plt.legend()
plt.grid()

plt.savefig(
    "results/final_loss_comparison.png"
)

plt.show()


print("Final graphs generated successfully.")