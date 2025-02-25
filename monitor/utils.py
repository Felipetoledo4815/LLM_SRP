import os
import pickle
import networkx as nx
from networkx.drawing import nx_pydot
from typing import Callable, Union
from functools import partial, lru_cache


class Node:
    def __init__(self, name, base_class=None):
        self.name = name
        self.base_class = base_class

    def __repr__(self):
        return str(self.name)   # name should always be a string anyway, but just to be safe


class IgnoreWaypointPickler(pickle.Pickler):
    def reducer_override(self, obj):
        """Custom reducer for MyClass."""
        if getattr(obj, "__name__", None) == "Waypoint":
            return None
        else:
            # For any other object, fallback to usual reduction
            return NotImplemented

    def persistent_id(self, obj):
        if str(type(obj)) == "<class 'carla.libcarla.Waypoint'>":
            return 'please_ignore_me'
        else:
            return None  # default behavior


class SGUnpickler(pickle.Unpickler):
    def persistent_load(self, pid):
        if pid == 'please_ignore_me':
            # Return a dummy object or None, depending on your use case
            return None
        else:
            # For any other persistent id, fallback to default behavior
            return super().persistent_load(pid)

    def find_class(self, module, name):
        if module == '__main__' and name == 'Node':
            return Node
        return super().find_class(module, name)


@lru_cache(maxsize=int(os.getenv('SG_CACHE_SIZE', default='128')))
def load_sg(sg_file):
    with open(sg_file, 'rb') as f:
        sg = SGUnpickler(f).load()
    return sg


def evaluate_predicate(predicate: Callable, sg: nx.DiGraph, func_chain=None) -> Union[bool, set]:
    if func_chain is None:
        func_chain = ''
    func_chain += '.' + predicate.func.__name__
    param_list = []
    for arg in predicate.args:
        if isinstance(arg, partial):
            param_list.append(evaluate_predicate(arg, sg, func_chain=func_chain))
        else:
            param_list.append(arg)
    if predicate.func.__name__ in ['filter_by_attr', 'rel_set']:
        return predicate.func(*param_list, sg, **predicate.keywords)
    else:
        return predicate.func(*param_list, **predicate.keywords)


def plot_graph(graph: nx.Graph, file_path: str):
    # Convert the NetworkX graph to a Pydot graph
    pydot_graph = nx_pydot.to_pydot(graph)
    
    # Save the Pydot graph to a file
    pydot_graph.write_png(file_path)