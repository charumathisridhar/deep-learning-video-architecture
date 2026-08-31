ACTION_CLASSES = [
    "WalkingWithDog",
    "Running",
    "JumpingJack",
    "Basketball",
    "PlayingGuitar"
]

NUM_CLASSES = len(ACTION_CLASSES)

print("Selected action classes:")
for i, action in enumerate(ACTION_CLASSES):
    print(f"{i}: {action}")

print("Number of classes:", NUM_CLASSES)