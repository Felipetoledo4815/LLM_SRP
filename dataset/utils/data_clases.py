from typing import Tuple
from scipy.spatial.transform import Rotation as R
from matplotlib.axes import Axes
import numpy as np


class Entity:
    def __init__(self, entity_type: str, xyz: np.ndarray, whl: np.ndarray, ypr: R) -> None:
        # Entity type
        self.entity_type = entity_type
        # Location
        self.xyz = xyz
        # Width, height, length
        self.whl = whl
        # Yaw, pitch, roll
        self.ypr = ypr

    def corners(self, whl_factor: float = 1.0) -> np.ndarray:
        """
        Returns the bounding box corners.
        :param wlh_factor: Multiply w, l, h by a factor to scale the box.
        :return: <np.float: 3, 8>. First four corners are the ones facing forward.
            The last four are the ones facing backwards.
        """
        w, h, l = self.whl * whl_factor

        # 3D bounding box corners. (Convention: x points forward, y to the left, z up.)
        x_corners = l / 2 * np.array([1,  1,  1,  1, -1, -1, -1, -1])
        y_corners = w / 2 * np.array([1, -1, -1,  1,  1, -1, -1,  1])
        z_corners = h / 2 * np.array([1,  1, -1, -1,  1,  1, -1, -1])
        corners = np.vstack((x_corners, y_corners, z_corners))

        # Rotate
        corners = np.dot(self.ypr.as_matrix(), corners)

        # Translate
        x, y, z = self.xyz
        corners[0, :] = corners[0, :] + x
        corners[1, :] = corners[1, :] + y
        corners[2, :] = corners[2, :] + z

        return corners

    def bottom_corners(self) -> np.ndarray:
        """
        Returns the four bottom corners.
        :return: <np.float: 3, 4>. Bottom corners. First two face forward, last two face backwards.
        """
        return self.corners()[:, [0, 1, 5, 4]]

    def render(self,
               axis: Axes,
               colors: Tuple = ('b', 'r', 'k'),
               linewidth: float = 2) -> None:
        """
        Renders the box in the provided Matplotlib axis.
        :param axis: Axis onto which the box should be drawn.
        :param colors: (<Matplotlib.colors>: 3). Valid Matplotlib colors (<str> or normalized RGB tuple) for front,
            back and sides.
        :param linewidth: Width in pixel of the box sides.
        """
        bottom_corners = self.bottom_corners()

        def draw_rect(selected_corners, color):
            prev = selected_corners[-1]
            for corner in selected_corners:
                axis.plot(
                    [prev[0], corner[0]],
                    [prev[2], corner[2]],
                    color=color, linewidth=linewidth, label=self.entity_type)
                prev = corner

        draw_rect(bottom_corners.T, colors)

        # Draw line indicating the front
        center_bottom_forward = np.mean(bottom_corners.T[0:2], axis=0)
        center_bottom = np.mean(bottom_corners.T, axis=0)
        axis.plot([center_bottom[0], center_bottom_forward[0]],
                  [center_bottom[2], center_bottom_forward[2]],
                  color=colors, linewidth=linewidth)

    def get_projected_center_point(self) -> np.ndarray:
        """
        Returns the center of the box in the x/y plane.
        :return: <np.float: 2>. The center of the box in (x, y).
        """
        bottom_corners = self.bottom_corners()
        return np.mean(bottom_corners.T, axis=0)[[0, 2]]


class EgoVehicle(Entity):
    def __init__(self, xyz: np.ndarray, whl: np.ndarray, ypr: R) -> None:
        super().__init__('ego', xyz, whl, ypr)

    def corners(self, whl_factor: float = 1.0) -> np.ndarray:
        """
        Returns the bounding box corners.
        :param wlh_factor: Multiply w, l, h by a factor to scale the box.
        :return: <np.float: 3, 8>. First four corners are the ones facing forward.
            The last four are the ones facing backwards.
        """
        w, h, l = self.whl * whl_factor

        # 3D bounding box corners. (Convention: x points forward, y to the left, z up.)
        x_corners = l / 2 * np.array([1,  1,  1,  1, -1, -1, -1, -1])
        y_corners = w / 2 * np.array([1, -1, -1,  1,  1, -1, -1,  1])
        z_corners = h / 2 * np.array([1,  1, -1, -1,  1,  1, -1, -1])
        corners = np.vstack((z_corners, y_corners, x_corners))

        return corners

    def bottom_corners(self) -> np.ndarray:
        """
        Returns the four bottom corners.
        :return: <np.float: 3, 4>. Bottom corners. First two face forward, last two face backwards.
        """
        return self.corners()[:, [1, 2, 6, 5]]
