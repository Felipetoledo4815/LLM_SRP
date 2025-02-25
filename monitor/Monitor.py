import os
import pandas as pd
import pickle
import sg_primitives as P
import utils
import Property
from monitor.properties0 import all_properties
from time import time
from PIL import Image
from functools import partial
from pathlib import Path


class Monitor:
    def __new__(cls, *args, **kwargs):
        if (len(args) > 0 or len(kwargs) > 0) and hasattr(cls,
                                                          'monitor_instance'):
            del cls.monitor_instance

        if not hasattr(cls, 'monitor_instance'):
            cls.monitor_instance = super(Monitor, cls).__new__(cls)
            cls.monitor_instance.initialize(*args, **kwargs)
        return cls.monitor_instance

    def initialize(
            self, properties:list=None,
            save_sg=False):
        self.properties = properties
        self.timestep = 0
        self.save_sg = save_sg

    def add_property(self, property):
        self.properties.append(property)

    def get_property(self, name):
        ret = [prop for prop in self.properties if prop.name == name]
        if len(ret) == 0:
            return None
        return ret[0]

    def save_all_relevant_subgraphs(self, sg, sg_name):
        save_dir = self.route_path / sg_name
        save_dir.mkdir(parents=True, exist_ok=True)
        for prop in self.properties:
            prop.save_relevant_subgraph(sg, str(save_dir/f'{prop.name}.png'))

    def get_all_relevant_subgraphs(self, sg):
        return {prop.name: prop.save_relevant_subgraph(sg, None) for prop in self.properties}

    def reset_properties(self):
        for i, prop in enumerate(self.properties):
            prop.reset()

    def check(self, sg, save_usage_information=False):
        # Save sg
        if self.save_sg:
            self.save_rsv(sg)

        results = {}

        # Check property
        for i, prop in enumerate(self.properties):
            # Update property data
            prop.update_data(sg, save_usage_information=save_usage_information)
            # Check property
            prop_eval, state = prop.check_step(return_state=True)
            if save_usage_information:
                sg.graph[f'state_{prop.name}'] = state
                sg.graph[f'acc_{prop.name}'] = prop_eval
            pred_eval = prop.get_last_predicates()

            pred_eval["property"] = prop_eval
            results[prop.name] = pred_eval

        self.timestep += 1
        return results

    def save_rsv(self, rsv):
        # Save pkl file
        with open(self.rsv_path / f"{self.timestep}.pkl", 'wb') as f:
            pickle.dump(rsv, f)

    def save_images(self, images):
        for i, image in enumerate(images):
            pil_img = Image.fromarray(image)
            pil_img.save(self.img_path / f"ts_{self.timestep}_img_{i}.jpg")

    def log(self, prop_name, pred_eval, prop_eval):
        csv_path = self.route_path / f"{prop_name}.csv"
        keys = ["ts"]
        keys.extend(list(pred_eval.keys()))
        keys.append("prop_eval")
        values = [self.timestep]
        values.extend(list(pred_eval.values()))
        values.append(prop_eval)
        if self.timestep > 0:
            # Load csv file
            df = pd.read_csv(csv_path)
            df.loc[len(df)] = values
            df.to_csv(csv_path, index=False)
        else:
            # Create new csv file
            df = pd.DataFrame(columns=keys)
            df.loc[0] = values
            df.to_csv(csv_path, index=False)


# Define a Node class
class Node:
    def __init__(self, name, speed=None, acceleration=None):
        self.name = name
        self.attr = {
            "speed": speed,
            "acceleration": acceleration
        }

    def __repr__(self):
        return f"Node(name={self.name})"

    def __eq__(self, other):
        return isinstance(other, Node) and self.name == other.name

    def __hash__(self):
        return hash(self.name)


### TEMPORARY
import networkx as nx
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


def main():
    m = Monitor()
    m.add_property(all_properties[0])

    triplet = [['bicycle', 'within_25m', 'ego'], ['bicycle', 'in_front_of', 'ego']]
    sg = triplets_to_nx_graph(triplet, 25, 1)

    m.check(sg)
    print("STOP")



if __name__ == '__main__':
    main()