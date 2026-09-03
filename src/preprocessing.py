import cv2
import os

# Dataset location
DATASET_PATH = "dataset"

# Where extracted frames will be saved
OUTPUT_PATH = "frames"

# Classes we are using
CLASSES = ["walk", "stand_up", "sit_down"]

# Number of frames extracted from each video
FRAMES_PER_VIDEO = 10


def extract_frames(video_path, output_folder):
    cap = cv2.VideoCapture(video_path)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames == 0:
        print("Could not read:", video_path)
        return

    # Select 10 equally spaced frames
    frame_numbers = [
        int(i * total_frames / FRAMES_PER_VIDEO)
        for i in range(FRAMES_PER_VIDEO)
    ]

    for index, frame_number in enumerate(frame_numbers):

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

        success, frame = cap.read()

        if success:
            filename = os.path.join(
                output_folder,
                f"frame_{index}.jpg"
            )

            cv2.imwrite(filename, frame)

    cap.release()


def process_dataset():

    for class_name in CLASSES:

        input_folder = os.path.join(DATASET_PATH, class_name)
        output_folder = os.path.join(OUTPUT_PATH, class_name)

        os.makedirs(output_folder, exist_ok=True)

        if not os.path.exists(input_folder):
            print("Folder not found:", input_folder)
            continue

        videos = os.listdir(input_folder)

        print(f"\nProcessing {class_name}...")

        for video_index, video_name in enumerate(videos):

            video_path = os.path.join(input_folder, video_name)

            video_output = os.path.join(
                output_folder,
                f"video_{video_index}"
            )

            os.makedirs(video_output, exist_ok=True)

            extract_frames(video_path, video_output)

            print(f"Processed: {video_name}")


if __name__ == "__main__":
    process_dataset()