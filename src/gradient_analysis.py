import tensorflow as tf

from models import build_shallow_cnn, build_deep_cnn


# Load one batch of training data
dataset = tf.keras.utils.image_dataset_from_directory(
    "frames",
    image_size=(128, 128),
    batch_size=16,
    shuffle=True,
    seed=42
)

images, labels = next(iter(dataset))


def calculate_gradient(model, images, labels):

    loss_function = tf.keras.losses.SparseCategoricalCrossentropy()

    with tf.GradientTape() as tape:

        predictions = model(images, training=True)

        loss = loss_function(labels, predictions)

    gradients = tape.gradient(
        loss,
        model.trainable_weights
    )

    gradient_values = []

    for gradient in gradients:

        if gradient is not None:
            gradient_values.append(
                tf.reduce_mean(tf.abs(gradient)).numpy()
            )

    average_gradient = sum(gradient_values) / len(gradient_values)

    return loss.numpy(), average_gradient


# --------------------------------
# Shallow CNN
# --------------------------------

shallow_model = build_shallow_cnn()

shallow_loss, shallow_gradient = calculate_gradient(
    shallow_model,
    images,
    labels
)


# --------------------------------
# Deep CNN
# --------------------------------

deep_model = build_deep_cnn()

deep_loss, deep_gradient = calculate_gradient(
    deep_model,
    images,
    labels
)


# --------------------------------
# Results
# --------------------------------

print("\n==============================")
print("TRAINING DIFFICULTY ANALYSIS")
print("==============================")

print(f"\nShallow CNN Loss     : {shallow_loss:.6f}")
print(f"Shallow CNN Gradient : {shallow_gradient:.8f}")

print(f"\nDeep CNN Loss        : {deep_loss:.6f}")
print(f"Deep CNN Gradient    : {deep_gradient:.8f}")


print("\n==============================")
print("INTERPRETATION")
print("==============================")

if deep_gradient < shallow_gradient:
    print(
        "Deep CNN shows smaller gradients, "
        "indicating potentially greater training difficulty."
    )
else:
    print(
        "Deep CNN does not show smaller gradients "
        "for this batch."
    )