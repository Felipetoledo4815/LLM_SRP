import tensorflow as tf
import os
import numpy as np
from waymo_open_dataset import dataset_pb2 as open_dataset
import tempfile
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import json
import pickle

waymo_db_folder = Path('/home/radowanredoy/Desktop/rotation1/lane_detection/waymo/v143/')
openlane_folder = Path('/home/radowanredoy/Desktop/rotation1/lane_detection/LATR/data/openlane/lane3d_1000/')
output_dir = Path('/home/radowanredoy/Desktop/rotation1/LLM_SRP/dboutput/')  # Use the current directory
data_type = 'validation'

def main():
    idx = 0
    tfrecord_list = []
    for file in os.listdir(waymo_db_folder):
        if file.endswith(".tfrecord"):
            tfrecord_list.append(waymo_db_folder / file)
    assert len(tfrecord_list) > 0, "No tfrecord files found in the dataset path."

    for file in tqdm(tfrecord_list, total=len(tfrecord_list)):
        print(file)
        dataset = tf.data.TFRecordDataset([file], compression_type='')
        dataset_iter = dataset.as_numpy_iterator()

        for data in dataset_iter:
            frame = open_dataset.Frame()
            frame.ParseFromString(data)

            timestamp_micros = frame.timestamp_micros * 100

            # Save the image
            img_path = '/home/radowanredoy/Desktop/rotation1/LLM_SRP/dboutput/image/' + f"img_{timestamp_micros}_{idx}.jpg"
            if not os.path.exists('/home/radowanredoy/Desktop/rotation1/LLM_SRP/dboutput/image/'):
                os.makedirs('/home/radowanredoy/Desktop/rotation1/LLM_SRP/dboutput/image/', exist_ok=True)
            if not os.path.exists('/home/radowanredoy/Desktop/rotation1/LLM_SRP/dboutput/pkl/'):
                os.makedirs('/home/radowanredoy/Desktop/rotation1/LLM_SRP/dboutput/pkl/', exist_ok=True)
            if not os.path.exists('/home/radowanredoy/Desktop/rotation1/LLM_SRP/dboutput/lane/'):
                os.makedirs('/home/radowanredoy/Desktop/rotation1/LLM_SRP/dboutput/lane/', exist_ok=True)
            with open(img_path, "wb") as f:
                f.write(frame.images[0].image)

            # Save labels
            labels_path = output_dir/ "pkl/" / f"labels_{timestamp_micros}_{idx}.pkl"
            label = {
                "laser_labels": [label for label in frame.laser_labels],
                "projected_lidar_labels": [label for label in frame.projected_lidar_labels],
                "context": frame.context
            }
            pickle.dump(label, open(labels_path, "wb"))

            # Save Lanes
            lane_path = output_dir / "lane/" / f"lane_{timestamp_micros}_{idx}.json"
            lane_data_json_path = openlane_folder / ('validation/' if data_type == 'validation' else '') / file.stem / f"{timestamp_micros}.json"
            if os.path.exists(lane_data_json_path):
                print("File exists.")
                with open(lane_data_json_path, 'r') as f:
                    data = json.load(f)
                    print(len(data['lane_lines']))
                    lane_label = {
                        "lane_label": data['lane_lines']
                    }
                pickle.dump(lane_label, open(lane_path, "wb"))
            else:
                print("File does not exist.")

            # Increase the data index
            idx += 1

        print(f"Processed {idx} frames...")
    return

if __name__ == "__main__":
    main()

