from typing import List
from PIL import Image
from matplotlib.axes import Axes
from matplotlib.patches import Arc
import matplotlib.pyplot as plt
import math

from dataset.utils.data_clases import Entity, EgoVehicle
from dataset.utils.color_map import get_color


class ScenePlot:
    def __init__(self, field_of_view) -> None:
        self.field_of_view = field_of_view

    def render_scene(self, ego_vehicle: EgoVehicle, entities: List[Entity], image_path: str,
                     out_path: None | str = None, title: None | str = None) -> None:

        fig, axs = plt.subplots(1, 2, figsize=(10, 5))

        data = Image.open(image_path)
        axs[0].imshow(data)

        for entity in entities:
            entity.render(axs[1], colors=get_color(entity.entity_type), linewidth=2)

        ego_vehicle.render(axs[1], colors=get_color(ego_vehicle.entity_type), linewidth=2)
        self.__plot_fov__(axs[1], ego_vehicle)
        self.__clean_legend__(axs[1])

        if title is not None:
            fig.suptitle(title)
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

    def __plot_fov__(self, ax: Axes, ego_vehicle: EgoVehicle) -> None:
        # Center point of the ego vehicle
        center_x, center_y = ego_vehicle.get_projected_center_point()

        # Orientation of the ego vehicle in radians
        orientation = math.radians(90)

        # Half of the FOV angle to calculate left and right FOV lines
        half_fov = math.radians(self.field_of_view / 2)

        # Third of the FOV angle to plot relative relationship thresholds
        third_fov = math.radians(self.field_of_view / 3 / 2)

        # Distance for the FOV lines from the center point
        distance = 50  # Adjust as needed

        for angle, color in [(half_fov, 'r--'), (third_fov, 'g--')]:
            # Calculate direction of the left FOV line
            left_fov_x = center_x + distance * math.cos(orientation - angle)
            left_fov_y = center_y + distance * math.sin(orientation - angle)

            # Calculate direction of the right FOV line
            right_fov_x = center_x + distance * math.cos(orientation + angle)
            right_fov_y = center_y + distance * math.sin(orientation + angle)

            # Plot the FOV lines
            ax.plot([center_x, left_fov_x], [center_y, left_fov_y], color)  # Left FOV line
            ax.plot([center_x, right_fov_x], [center_y, right_fov_y], color)  # Right FOV line

        for arc_radius in [10, 25, 50]:
            # Arc parameters
            start_angle = math.degrees(orientation - half_fov)  # Start angle in degrees
            end_angle = math.degrees(orientation + half_fov)  # End angle in degrees

            # Create and add the arc to the plot
            arc = Arc((center_x, center_y), 2*arc_radius, 2*arc_radius,
                    angle=0, theta1=start_angle, theta2=end_angle, edgecolor='r', linestyle=':', lw=2)
            ax.add_patch(arc)
