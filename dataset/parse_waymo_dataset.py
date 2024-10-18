import argparse
import os
from pathlib import Path
import pickle
import tensorflow as tf
from waymo_open_dataset import dataset_pb2 as open_dataset
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(description="Parse Waymo dataset")
    parser.add_argument("--dataset_path", type=Path, help="Path to the Waymo dataset")
    parser.add_argument("--output_path", type=Path, help="Path to the output directory")
    parser.add_argument("--remove_tfrecord", action="store_true", help="Remove tfrecord files after parsing.")
    args = parser.parse_args()

    # Check if the output directory exists
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    # Data index
    idx = 0

    # Get tfrecord files
    tfrecord_list = []
    for file in os.listdir(args.dataset_path):
        if file.endswith(".tfrecord"):
            tfrecord_list.append(args.dataset_path / file)
    assert len(tfrecord_list) > 0, "No tfrecord files found in the dataset path."

    # Load the dataset
    for file in tqdm(tfrecord_list, total=len(tfrecord_list)):

        dataset = tf.data.TFRecordDataset(args.dataset_path / file, compression_type='')
        dataset_iter = dataset.as_numpy_iterator()

        for data in dataset_iter:
            frame = open_dataset.Frame()
            frame.ParseFromString(data)

            # Save the image
            img_path = str(args.output_path / f"img_{idx}.jpg")
            tf.io.write_file(img_path, frame.images[0].image)

            # Save labels
            labels_path = args.output_path / f"labels_{idx}.pkl"
            label = {
                "laser_labels": [label for label in frame.laser_labels],
                "projected_lidar_labels": [label for label in frame.projected_lidar_labels],
                "context": frame.context
            }
            pickle.dump(label, open(labels_path, "wb"))

            # Increase the data index
            idx += 1

        print(f"Processed {idx} frames...")

        # Remove the tfrecord file
        if args.remove_tfrecord:
            os.remove(args.dataset_path / file)


if __name__ == "__main__":
    main()
