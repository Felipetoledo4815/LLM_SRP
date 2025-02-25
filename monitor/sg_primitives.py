import networkx as nx
import re
import math
from typing import Union, Callable
from functools import partial


def parse_node_set(node_set: Union[set, str], sg: nx.DiGraph) -> set:
    """ Parse node_set if it is a string.
    :param node_set: Set of nodes. It can be a set of networkx nodes
        or a string. The string must be either "Ego" (Set containing the Ego
        car node) or "G" (Set containing all nodes in Graph).
    :param sg: Networkx Directed Scene Graph.
    :return: Set of nodes.
    """
    assert isinstance(node_set, set) or isinstance(node_set, str), f"Invalid \
        node_set: {node_set}. It must be either a set of networkx nodes or a \
        string."
    if isinstance(node_set, str):
        assert node_set in ["Ego", "G"], f"Invalid node_set string: {node_set}\
            . It must be either 'Ego' or 'G'."
        new_node_set = set()
        if node_set == "Ego":
            for node in sg.nodes:
                if node.name == "ego":
                    new_node_set.add(node)
                    break
        elif node_set == "G":
            for node in sg.nodes:
                new_node_set.add(node)
        return new_node_set
    else:
        return node_set


def filter_by_attr(
        node_set: Union[set, str],
        attr: str,
        filter: Union[str, Callable[[float], bool]],
        sg: nx.DiGraph) -> set:
    """ Filter a set of nodes by attribute value.
    :param node_set: Set of nodes to filter. It can be a set of networkx nodes
        or a string. The string must be either "Ego" (Set containing the Ego
        car node) or "G" (Set containing all nodes in Graph).
    :param attr: Attribute name to filter by. String representing the name of
        the attribute to filter by, e.g. "name".
    :param filter: Filter to apply to attribute values. It can be a string
        representing a regular expression that filters the node attribute,
        e.g. "Lane*". Or it can be a comparisson function for numeric
        attributes, e.g. greater_than(1.5).
    :param sg: Networkx Directed Scene Graph.
    :return: Set of nodes with the given attribute values.
    """
    node_set = parse_node_set(node_set, sg)
    assert isinstance(node_set, set) or isinstance(node_set, str), f"Invalid \
        node_set: {node_set}. It must be either a set of networkx nodes or a \
        string."
    assert isinstance(filter, str) or callable(filter), f"Invalid filter: \
        {filter}. It must be either a string or a function."
    new_node_set = set()
    for node in node_set:
        if attr != "name" and attr != "base_class":
            node_attr = node.attr[attr]
        else:
            node_attr = getattr(node, attr)
        if isinstance(filter, str):
            if re.search(filter, node_attr):
                new_node_set.add(node)
        else:
            if filter(node_attr):
                new_node_set.add(node)
    return new_node_set


def rel_set(node_set: Union[set, str], rel: str, sg: nx.DiGraph,
           edge_type: str = "outgoing") -> set:
    """ Get the set of nodes with a given relationship
    :param node_set: Set of nodes to filter. It can be a set of networkx nodes
        or a string. The string must be either "Ego" (Set containing the Ego
        car node) or "G" (Set containing all nodes in Graph).
    :param rel: Relationship to filter by. String representing the name of
        the relationship to filter by, e.g. "isIn".
    :param edge_type: incoming or outgoing edges. Default: outgoing.
    :param sg: Networkx Directed Scene Graph.
    :return: Set of nodes with the given relationship.
    """
    node_set = parse_node_set(node_set, sg)
    assert edge_type in ["incoming", "outgoing"], f"Invalid edge_type: \
        {edge_type}. It must be either 'incoming' or 'outgoing'. Default: \
        outgoing."
    new_node_set = set()
    if edge_type == "outgoing":
        get_edges = partial(sg.out_edges, data="label")
    else:
        get_edges = partial(sg.in_edges, data="label")
    for node in node_set:
        for src, dst, edge in get_edges(node):
            if rel == edge:
                if edge_type == "outgoing":
                    new_node_set.add(dst)
                else:
                    new_node_set.add(src)
    return new_node_set


def union(s1: set, s2: set) -> set:
    return s1.union(s2)


def intersection(s1: set, s2: set) -> set:
    return s1.intersection(s2)


def difference(s1: set, s2: set) -> set:
    # TODO: check if this is the correct difference. Different based on order
    return s1.difference(s2)


def size(s: set) -> int:
    return len(s)


def lt(a, b):
    return a < b


def gt(a, b):
    return a > b


def le(a, b):
    return a <= b


def ge(a, b):
    return a >= b


def eq(a, b):
    return a == b


def ne(a, b):
    return a != b


def logic_or(a, b):
    return a or b


def logic_and(a, b):
    return a and b


def logic_not(a):
    return not a


def get_distance(node_set, ego_set):
    if len(ego_set) == 1:
        ego_node = list(ego_set)[0]
        x2, y2, z2 = ego_node.attr["vertex"]
    else:
        raise ValueError("Ego set must contain only one node")
    if len(node_set) == 0:
        return -1000
    distances = []
    for node in node_set:
        x1, y1, z1 = node.attr["vertex"]
        distances.append(math.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2))
    return min(distances)
