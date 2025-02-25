import pickle
import json
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import copy
import shutil

from pathlib import Path
from tqdm import tqdm
from properties import all_properties
from Monitor import Monitor


# Define a Node class
class Node:
    def __init__(self, name, speed=None, acceleration=None):
        self.name = name
        # self.speed = speed
        # self.acceleration = acceleration
        self.attr = {"speed": speed, "acceleration": acceleration}

    def __repr__(self):
        return f"Node(name={self.name})"

    def __eq__(self, other):
        return isinstance(other, Node) and self.name == other.name

    def __hash__(self):
        return hash(self.name)


def get_label_path(img_path):
    label_path = img_path.replace("img_", "labels_")
    label_path = label_path.replace(".jpg", ".pkl")
    label_path = label_path.replace("/waymo_1.4/", "./monitor/labels/")
    return label_path


def triplets_to_nx_graph(triplets, speed, acceleration):
    g = nx.MultiDiGraph()
    nodes = {}

    # Add edges to the graph with unique identifiers for 'vehicle'
    for i, (source, relation, target) in enumerate(triplets):
        unique_source = Node(f"{source}_{i}")
        target_node = Node(target, speed, acceleration)

        # Add nodes with the "name" attribute
        if unique_source not in nodes:
            g.add_node(unique_source, name=unique_source.name)
            nodes[unique_source] = unique_source
        if target_node not in nodes:
            g.add_node(target_node, name=target_node.name)
            nodes[target_node] = target_node

        # Add edge with the relation attribute
        g.add_edge(unique_source, target_node, label=relation)

    return g


def calculate_speed(pose_f0, pose_f1):
    # pose_f0 = np.reshape(np.array(f0["pose"].transform), [4, 4])
    position_f0 = pose_f0[:, 3]
    # pose_f1 = np.reshape(np.array(f1["pose"].transform), [4, 4])
    position_f1 = pose_f1[:, 3]

    # The frequency is 10Hz, thus the delta between frames is 0.1 seconds

    ### Km/H
    # # To convert to Km/H multiply by 3600 seconds, and divide by 1000 meters
    # v = abs((position_f1 - position_f0) * 3600 / (0.1 * 1000))

    ### Mi/H
    # To convert to Mi/H multiply by 3600 seconds, and divide by 1609,34 meters
    v = abs((position_f1 - position_f0) * 3600 / (0.1 * 1609.34))

    return v


def calculate_acc(v1, v2):
    # The frequency is 10Hz, thus the delta between frames is 0.1 seconds

    ### Km/H
    # # To convert to m/s^2 from Km/H multiply by 1000 meters, and divide by 3600 seconds
    # acc = ((v2 - v1) * 1000 / 3600) / 0.1

    ### Mi/H
    # To convert to m/s^2 from Mi/H multiply by 1609,34 meters, and divide by 3600 seconds
    acc = ((v2 - v1) * 1609.34 / 3600) / 0.1
    return acc


def plot_speed_and_acceleration(speed, acceleration):
    # Sample data
    time = [i for i in range(len(speed))]  # Time in seconds

    # Create a figure and axis
    plt.figure()

    # Plot speed
    plt.plot(time, speed, label="Speed", color="blue")

    # Plot acceleration
    plt.plot(time, acceleration, label="Acceleration", color="red")
    plt.plot(time, [0 for i in range(len(acceleration))])

    # Add labels and title
    plt.xlabel("Time (s)")
    plt.ylabel("Value")
    plt.title("Speed and Acceleration over Time")

    # Add a legend
    plt.legend()

    # Show the plot
    plt.show()


def main():

    all_properties_copy = copy.deepcopy(all_properties)
    
    ### Load the log files
    log_dirs = [
        "./monitor/val_metrics.json",
        "./monitor/train_metrics.json"
    ]
    log = []
    for log_dir in log_dirs:
        with open(log_dir, "r") as file:
            log.extend(json.load(file))

    # Sort log list based on image_id
    sorted_log = sorted(log, key=lambda x: x["image_id"])

    # Create logger
    logger = Logger("./results/")

    gt_m = None
    all_seq_summary = None
    n_sequences = 0
    pbar = tqdm(total=len(sorted_log), desc="Processing frames")
    local_idx = -1

    for idx in range(0, len(sorted_log)):
        # Load .pkl file
        lbl_p = get_label_path(sorted_log[idx]["img_path"])
        with open(lbl_p, "rb") as f:
            frame = pickle.load(f)

        # Sequence start
        if frame["start_frame"] == 1:
            n_sequences += 1

            local_idx = 0
            poses = [np.reshape(np.array(frame["pose"].transform), [4, 4])]
            if gt_m is not None:
                # Log sequence summary
                seq_summary = logger.log_seq_summary(results_ground_truth, results_prediction, n_sequences-1)
                if all_seq_summary is None:
                    all_seq_summary = seq_summary
                else:
                    all_seq_summary += seq_summary
                
                # Reset properties in monitor
                gt_m.reset_properties()
                pred_m.reset_properties()
            else:
                # Create monitors
                gt_m = Monitor(all_properties)
                pred_m = Monitor(all_properties_copy)
                
            # # Update the progress bar
        elif local_idx >= 0:
            local_idx += 1
            poses.append(np.reshape(np.array(frame["pose"].transform), [4, 4]))

        # After first 2 frames (Need to compute speed & acc)
        if local_idx > 1:
            v1 = calculate_speed(poses[-3], poses[-2])
            # Current frame speed and acc
            v2 = calculate_speed(poses[-2], poses[-1])
            acc = calculate_acc(v1, v2)

            # Scene Graphs
            ground_truth_sg = triplets_to_nx_graph(
                sorted_log[idx]["ground_truth"], v2[0], acc[0]
            )
            prediction_sg = triplets_to_nx_graph(
                sorted_log[idx]["prediction"], v2[0], acc[0]
            )

            # Monitor
            results_ground_truth = gt_m.check(ground_truth_sg)
            results_prediction = pred_m.check(prediction_sg)
            
            # Add image path and local idx from the sequence
            additional_info = {
                "img_path": sorted_log[idx]["img_path"],
                "ts": local_idx,
                "speed": round(v2[0], 3),
                "acc": round(acc[0], 3)
            }
            results_ground_truth = {**additional_info, "properties":results_ground_truth}
            results_prediction = {**additional_info, "properties":results_prediction}

            logger.log_results(results_ground_truth, n_sequences, True)
            logger.log_results(results_prediction, n_sequences, False)
            
        # Update the progress bar
        pbar.update(1)
    
    # Log sequence summary
    if gt_m is not None:
        seq_summary = logger.log_seq_summary(results_ground_truth, results_prediction, n_sequences-1)
        if all_seq_summary is None:
            all_seq_summary = seq_summary
        else:
            all_seq_summary += seq_summary

    pbar.close()
    logger.log_final_summary(all_seq_summary)


class Logger():
    def __init__(self, log_folder_path):
        self.log_path = Path(log_folder_path)
        self.log_path.mkdir(parents=True, exist_ok=True)

    def log_results(self, results, n_sequences, ground_truth=False):
        for prop_name, propositions in results["properties"].items():
            keys = ["img_path", "ts", "speed", "acc"]
            keys.extend(list(propositions.keys()))
            values = [results["img_path"], results["ts"], results["speed"], results["acc"]]
            values.extend(list(propositions.values()))

            csv_path = self.log_path / f"seq_{n_sequences}"
            csv_path.mkdir(parents=True, exist_ok=True)
            result_type = 'gt' if ground_truth else 'pred'
            csv_path = csv_path / f"{prop_name}_{result_type}.csv"

            if results["ts"] == 2:
                # Create new csv file
                df = pd.DataFrame(columns=keys)
                df.loc[0] = values
                df.to_csv(csv_path, index=False)
            else:
                # Load csv file
                df = pd.read_csv(csv_path)
                df.loc[len(df)] = values
                df.to_csv(csv_path, index=False)

    def log_seq_summary(self, results_ground_truth, results_prediction, n_sequences):
        seq_summary = {}
        for name, data in results_ground_truth["properties"].items():
            seq_summary[name] = [data["property"]]
        for name, data in results_prediction["properties"].items():
            seq_summary[name].append(data["property"])
        df = pd.DataFrame(seq_summary)
        df.index = ["Ground Truth", "Prediction"]
        df = df.T
        df = (df.astype(int) - 1) * -1
        # Create new csv file
        csv_path = self.log_path / f"seq_{n_sequences}" / f"seq_{n_sequences}_summary.csv"
        df.to_csv(csv_path)

        return df
    
    def log_final_summary(self, all_seq_summary):
        # Create new csv file
        csv_path = self.log_path / "all_summary.csv"
        all_seq_summary.to_csv(csv_path)


if __name__ == "__main__":
    main()