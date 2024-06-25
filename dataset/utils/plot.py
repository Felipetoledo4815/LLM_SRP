from typing import List
from PIL import Image
from matplotlib.axes import Axes
import matplotlib.pyplot as plt

from dataset.utils.data_clases import Entity, EgoVehicle
from dataset.utils.color_map import get_color


class ScenePlot:
    def __init__(self) -> None:
        pass

    def render_scene(self,
                     ego_vehicle: EgoVehicle,
                     entities: List[Entity],
                     image_path: str,
                     out_path: str = '') -> None:
        _, axs = plt.subplots(1, 2, figsize=(10, 5))

        data = Image.open(image_path)
        axs[0].imshow(data)

        for entity in entities:
            entity.render(axs[1], colors=get_color(entity.entity_type), linewidth=2)

        ego_vehicle.render(axs[1], colors=get_color(ego_vehicle.entity_type), linewidth=2)
        self.__clean_legend__(axs[1])

        plt.tight_layout()

        if out_path is not None:
            plt.savefig(out_path, bbox_inches='tight', pad_inches=0, dpi=200)
        else:
            plt.show()

    def __clean_legend__(self, ax: Axes) -> None:
        # Collect all labels and handles
        handles, labels = ax.get_legend_handles_labels()

        # Filter out duplicate labels and handles
        new_labels, new_handles = [], []
        for handle, label in zip(handles, labels):
            if label not in new_labels:
                new_labels.append(label)
                new_handles.append(handle)

        # Create the legend with filtered labels and handles
        ax.legend(new_handles, new_labels, loc="best")
