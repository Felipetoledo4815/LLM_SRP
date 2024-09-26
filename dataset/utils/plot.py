from typing import List
import math
import matplotlib.pyplot as plt
import numpy as np
import statistics
from PIL import Image
from matplotlib.axes import Axes
from matplotlib import patches
from dataset.utils.data_clases import Entity, EgoVehicle, EntityType


class ScenePlot:
    def __init__(self, field_of_view) -> None:
        self.field_of_view = field_of_view

    def render_scene(self, ego_vehicle: EgoVehicle, entities: List[Entity], image_path: str,
                     out_path: None | str = None, title: None | str = None) -> None:

        fig, axs = plt.subplots(1, 2, figsize=(10, 5))

        data = Image.open(image_path)
        axs[0].imshow(data)
        axs[0].axis('off')

        for entity in entities:
            entity.render(axs[1], colors=entity.get_color(), linewidth=2)

        ego_vehicle.render(axs[1], colors=ego_vehicle.get_color(), linewidth=2)
        self.__plot_fov__(axs[1], ego_vehicle)
        self.__clean_legend__(axs[1])

        if title is not None:
            fig.suptitle(title)
        plt.tight_layout()

        # Adjust the x and y axis
        range_x = axs[1].get_xticks()
        median_x = statistics.median(range_x)
        x_range = range_x[-1] - range_x[0]
        range_y = axs[1].get_yticks()
        median_y = statistics.median(range_y)
        y_range = range_y[-1] - range_y[0]
        final_range = max(x_range, y_range)
        axs[1].set_xlim(median_x - final_range / 2, median_x + final_range / 2)
        axs[1].set_ylim(median_y - final_range / 2, median_y + final_range / 2)

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
        distance = 60  # Adjust as needed

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

        for arc_radius in [25, 40, 60]:  # Radius based on distance thresholds
            # Arc parameters
            start_angle = math.degrees(orientation - half_fov)  # Start angle in degrees
            end_angle = math.degrees(orientation + half_fov)  # End angle in degrees

            # Create and add the arc to the plot
            arc = patches.Arc((center_x, center_y), 2*arc_radius, 2*arc_radius,
                              angle=0, theta1=start_angle, theta2=end_angle, edgecolor='r', linestyle=':', lw=2)
            ax.add_patch(arc)

    def plot_2d_bounding_boxes(self, entities: List[Entity], image_path: str, out_path: None | str = None,
                               title: None | str = None) -> None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))

        data = Image.open(image_path)
        data_array = np.array(data)
        ax.imshow(data_array)
        ax.axis('off')

        for entity in entities:
            print(entity.entity_type)
            entity.render_bounding_box(ax, colors=entity.get_color(), linewidth=1)

        if title is not None:
            fig.suptitle(title)
        plt.tight_layout()

        if out_path is not None:
            plt.savefig(out_path, bbox_inches='tight', pad_inches=0, dpi=200)
        else:
            plt.show()

    def plot_2d_bounding_boxes_from_corners(self, bbs: List[str], image_path: str,
                                            out_path: None | str = None,
                                            title: None | str = None,
                                            entity_types: List[str] | None = None) -> None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))

        data = Image.open(image_path)
        data_array = np.array(data)
        ax.imshow(data_array)

        if entity_types is None:
            for bb in bbs:
                self.__render_bounding_box_from_corners(bb, ax, linewidth=1)
        else:
            for bb, entity_type in zip(bbs, entity_types):
                self.__render_bounding_box_from_corners(bb, ax, entity_type, linewidth=1)

        if title is not None:
            fig.suptitle(title)
        plt.tight_layout()

        if out_path is not None:
            plt.savefig(out_path, bbox_inches='tight', pad_inches=0, dpi=200)
        else:
            plt.show()

    def __render_bounding_box_from_corners(self, bb: str, axis: Axes,
                                           entity_type: str | None = None,
                                           linewidth: float = 1) -> None:
        """
        Renders the bounding box in the provided Matplotlib axis.
        :param bb: Bounding box in the format '(x1, y1, x2, y2)'.
        :param axis: Axis onto which the box should be drawn.
        :param entity_type: Type of the entity to retrieve its color.
        :param linewidth: Width in pixel of the box sides.
        """
        bb_list = bb.strip('()').split(',')
        bb_int_list = tuple(int(x) for x in bb_list)
        bottom_left = bb_int_list[0], bb_int_list[1]
        width = bb_int_list[2] - bb_int_list[0]
        height = bb_int_list[3] - bb_int_list[1]

        color = np.array([0, 0, 0])
        if entity_type is not None:
            color = EntityType.from_str(entity_type).color

        rect1 = patches.Rectangle(bottom_left, width, height, linewidth=linewidth, edgecolor=color, facecolor='none')
        axis.add_patch(rect1)
