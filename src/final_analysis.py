import tensorflow as tf
import matplotlib.pyplot as plt
import os

from models import (
    build_shallow_cnn,
    build_deep_cnn,
    build_improved_cnn
)


# --------------------------------
# Create results folder
# --------------------------------

os.makedirs("results", exist_ok=True)


# --------------------------------
# Create models
# --------------------------------

shallow_model = build_shallow_cnn()
deep_model = build_deep_cnn()
improved_model = build_improved_cnn()


# --------------------------------
# Get parameter counts
# --------------------------------

shallow_params = shallow_model.count_params()
deep_params = deep_model.count_params()
improved_params = improved_model.count_params()


# --------------------------------
# Approximate parameter memory
# --------------------------------

shallow_memory = (shallow_params * 4) / (1024 ** 2)
deep_memory = (deep_params * 4) / (1024 ** 2)
improved_memory = (improved_params * 4) / (1024 ** 2)


# --------------------------------
# Print comparison
# --------------------------------

print("\n==========================================")
print("FINAL ARCHITECTURAL COMPARISON")
print("==========================================")

print("\nModel                 Parameters      Memory")
print("-" * 50)

print(
    f"Shallow CNN          {shallow_params:>10,}      "
    f"{shallow_memory:.2f} MB"
)

print(
    f"Deep CNN             {deep_params:>10,}      "
    f"{deep_memory:.2f} MB"
)

print(
    f"Improved CNN         {improved_params:>10,}      "
    f"{improved_memory:.2f} MB"
)


# --------------------------------
# Parameter comparison graph
# --------------------------------

models = [
    "Shallow CNN",
    "Deep CNN",
    "Improved CNN"
]

parameters = [
    shallow_params,
    deep_params,
    improved_params
]

plt.figure(figsize=(8, 5))

plt.bar(models, parameters)

plt.xlabel("Architecture")
plt.ylabel("Number of Parameters")
plt.title("Parameter Comparison")

plt.savefig(
    "results/parameter_comparison.png"
)

plt.show()


# --------------------------------
# Memory comparison graph
# --------------------------------

memory = [
    shallow_memory,
    deep_memory,
    improved_memory
]

plt.figure(figsize=(8, 5))

plt.bar(models, memory)

plt.xlabel("Architecture")
plt.ylabel("Approximate Memory (MB)")
plt.title("Memory Requirement Comparison")

plt.savefig(
    "results/memory_comparison.png"
)

plt.show()


# --------------------------------
# Final conclusion
# --------------------------------

print("\n==========================================")
print("CONCLUSION")
print("==========================================")

print(
    "\nIncreasing network depth increases the "
    "number of parameters and computational requirements."
)

print(
    "Batch Normalization helps stabilize training, "
    "while Dropout helps reduce overfitting."
)

print(
    "Therefore, deeper architectures should be designed "
    "carefully according to available computational resources."
)