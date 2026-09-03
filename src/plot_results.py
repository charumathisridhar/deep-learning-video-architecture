import matplotlib.pyplot as plt

from train import (
    shallow_history,
    deep_history,
    shallow_time,
    deep_time
)


# -----------------------------
# Accuracy Comparison
# -----------------------------

plt.figure(figsize=(8, 5))

plt.plot(
    shallow_history.history["accuracy"],
    label="Shallow CNN - Training"
)

plt.plot(
    shallow_history.history["val_accuracy"],
    label="Shallow CNN - Validation"
)

plt.plot(
    deep_history.history["accuracy"],
    label="Deep CNN - Training"
)

plt.plot(
    deep_history.history["val_accuracy"],
    label="Deep CNN - Validation"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Shallow CNN vs Deep CNN - Accuracy")

plt.legend()
plt.grid()

plt.savefig("results/accuracy_comparison.png")

plt.show()


# -----------------------------
# Loss Comparison
# -----------------------------

plt.figure(figsize=(8, 5))

plt.plot(
    shallow_history.history["loss"],
    label="Shallow CNN - Training"
)

plt.plot(
    shallow_history.history["val_loss"],
    label="Shallow CNN - Validation"
)

plt.plot(
    deep_history.history["loss"],
    label="Deep CNN - Training"
)

plt.plot(
    deep_history.history["val_loss"],
    label="Deep CNN - Validation"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Shallow CNN vs Deep CNN - Loss")

plt.legend()
plt.grid()

plt.savefig("results/loss_comparison.png")

plt.show()


# -----------------------------
# Training Time Comparison
# -----------------------------

models = ["Shallow CNN", "Deep CNN"]
times = [shallow_time, deep_time]

plt.figure(figsize=(7, 5))

plt.bar(models, times)

plt.xlabel("Model")
plt.ylabel("Training Time (seconds)")
plt.title("Training Time Comparison")

plt.savefig("results/training_time_comparison.png")

plt.show()


print("\nResults saved in results folder.")