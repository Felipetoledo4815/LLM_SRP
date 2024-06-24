from typing import Dict, Tuple
import numpy as np


def get_colormap() -> Dict[str, Tuple[int, int, int]]:
    """
    Get the defined colormap.
    :return: A mapping from the class names to the respective RGB values.
    """

    classname_to_color = {  # RGB.
        "person": (0, 0, 230),  # Blue
        "bicycle": (220, 20, 60),  # Crimson
        "bus": (255, 127, 80),  # Coral
        "car": (255, 158, 0),  # Orange
        "construction": (233, 150, 70),  # Darksalmon
        "emergency vehicle": (255, 215, 0),  # Gold
        "motorcycle": (255, 61, 99),  # Red
        "trailer truck": (255, 140, 0),  # Darkorange
        "truck": (255, 99, 71),  # Tomato
        "ego": (0, 0, 0)    # Black
    }

    return classname_to_color

def get_color(class_name: str) -> Tuple[float, float, float]:
    colormap = get_colormap()
    c = np.array(colormap[class_name]) / 255.0
    return tuple(c)
