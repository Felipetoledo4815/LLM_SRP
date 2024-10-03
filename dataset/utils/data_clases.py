from typing import Tuple, List
from enum import Enum, auto
from scipy.spatial.transform import Rotation as R
from matplotlib import patches
from matplotlib.axes import Axes
import numpy as np

class EntityType(Enum):
    PERSON = (0, 0, 230)  # Blue
    BICYCLE = (220, 20, 60)  # Crimson
    BUS = (255, 127, 80)  # Coral
    CAR = (255, 158, 0)  # Orange
    CONSTRUCTION_VEHICLE = (233, 150, 70)  # Darksalmon
    EMERGENCY_VEHICLE = (255, 215, 0)  # Gold
    MOTORCYCLE = (255, 61, 99)  # Red
    TRAILER_TRUCK = (255, 140, 0)  # Darkorange
    TRUCK = (255, 99, 71)  # Tomato
    EGO = (0, 0, 0)  # Black

    @property
    def color(self) -> np.ndarray:
        return np.array(self.value) / 255.0

    @property
    def type_name(self) -> str:
        return self.name.lower()

    @classmethod
    def get_types(cls) -> List[str]:
        return [name.lower() for name, _ in cls.__members__.items()]

    @classmethod
    def from_str(cls, entity_type: str) -> 'EntityType':
        assert entity_type.lower() in cls.get_types(), f"Entity {entity_type} not recognized. Please add it to EntityType."
        return cls[entity_type.upper()]


class RelationshipType(Enum):
    WITHIN_25M = auto()
    BETWEEN_25M_AND_40M = auto()
    BETWEEN_40M_AND_60M = auto()
    IN_FRONT_OF = auto()
    TO_LEFT_OF = auto()
    TO_RIGHT_OF = auto()

    @property
    def type_name(self) -> str:
        return self.name.lower()

    @classmethod
    def get_types(cls) -> List[str]:
        return [name.lower() for name, _ in cls.__members__.items()]

    @classmethod
    def from_str(cls, relationship_type: str) -> 'RelationshipType':
        assert relationship_type in cls.get_types(
        ), f"Relationship {relationship_type} not recognized. Please add it to RelationshipType."
        return cls[relationship_type.upper()]


class Entity:
    def __init__(self, entity_type: str, xyz: np.ndarray, whl: np.ndarray, ypr: R,
                 camera_intrinsic: np.ndarray = np.eye(3)) -> None:
        # Entity type
        self.entity_type = EntityType.from_str(entity_type).type_name
        # Location
        self.xyz = xyz
        # Width, height, length
        self.whl = whl
        # Yaw, pitch, roll
        self.ypr = ypr
        # Camera intrinsic matrix
        self.camera_intrinsic = camera_intrinsic

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
        return self.corners()[:, [2, 3, 7, 6]]

    def render(self,
               axis: Axes,
               colors: np.ndarray,
               linewidth: float = 2) -> None:
        """
        Renders the box in the provided Matplotlib axis.
        :param axis: Axis onto which the box should be drawn.
        :param colors: (<Matplotlib.colors>: 3). Valid Matplotlib colors (<str> or normalized RGB tuple) for front,
            back and sides.
        :param linewidth: Width in pixel of the box sides.
        """
        bottom_corners = self.bottom_corners()
        polygon = patches.Polygon(bottom_corners[:2, :].T, closed=True, linewidth=1, edgecolor=colors,
                                  facecolor=(1.0, 1.0, 1.0, 0.0), label=self.entity_type)
        axis.add_patch(polygon)

        # Draw line indicating the front
        center_bottom_forward = np.mean(bottom_corners.T[0:2], axis=0)
        center_bottom = np.mean(bottom_corners.T, axis=0)
        axis.plot([center_bottom[0], center_bottom_forward[0]],
                  [center_bottom[1], center_bottom_forward[1]],
                  color=colors, linewidth=linewidth)

    def render_bounding_box(self, axis: Axes, colors: np.ndarray, linewidth: float = 1) -> None:
        """
        Renders the bounding box in the provided Matplotlib axis.
        :param axis: Axis onto which the box should be drawn.
        :param colors: (<Matplotlib.colors>: 3). Valid Matplotlib colors (<str> or normalized RGB tuple) for front,
            back and sides.
        :param linewidth: Width in pixel of the box sides.
        """
        bb = self.get_2d_bounding_box()
        bottom_left = bb[0], bb[1]
        width = bb[2] - bb[0]
        height = bb[3] - bb[1]
        rect1 = patches.Rectangle(bottom_left, width, height, linewidth=linewidth, edgecolor=colors, facecolor='none')
        axis.add_patch(rect1)

    def get_projected_center_point(self) -> np.ndarray:
        """
        Returns the center of the box in the x/y plane.
        :return: <np.float: 2>. The center of the box in (x, y).
        """
        bottom_corners = self.bottom_corners()
        return np.mean(bottom_corners.T, axis=0)[[0, 1]]

    def view_points(self, points: np.ndarray, view: np.ndarray, normalize: bool) -> np.ndarray:
        """
        This is a helper class that maps 3d points to a 2d plane. It can be used to implement both perspective and
        orthographic projections. It first applies the dot product between the points and the view. By convention,
        the view should be such that the data is projected onto the first 2 axis. It then optionally applies a
        normalization along the third dimension.

        For a perspective projection the view should be a 3x3 camera matrix, and normalize=True
        For an orthographic projection with translation the view is a 3x4 matrix and normalize=False
        For an orthographic projection without translation the view is a 3x3 matrix (optionally 3x4 with last columns
        all zeros) and normalize=False

        :param points: <np.float32: 3, n> Matrix of points, where each point (x, y, z) is along each column.
        :param view: <np.float32: n, n>. Defines an arbitrary projection (n <= 4).
            The projection should be such that the corners are projected onto the first 2 axis.
        :param normalize: Whether to normalize the remaining coordinate (along the third axis).
        :return: <np.float32: 3, n>. Mapped point. If normalize=False, the third coordinate is the height.
        """

        assert view.shape[0] <= 4
        assert view.shape[1] <= 4
        assert points.shape[0] == 3

        viewpad = np.eye(4)
        viewpad[:view.shape[0], :view.shape[1]] = view

        nbr_points = points.shape[1]

        # Do operation in homogenous coordinates.
        points = np.concatenate((points, np.ones((1, nbr_points))))
        points = np.dot(viewpad, points)
        points = points[:3, :]

        if normalize:
            points = points / points[2:3, :].repeat(3, 0).reshape(3, nbr_points)

        return points

    def get_2d_bounding_box(self) -> Tuple[int, int, int, int]:
        """
        Returns the 2D bounding box of the entity.
        """
        raise NotImplementedError("Subclass must implement this method")

    def get_color(self) -> np.ndarray:
        """
        Returns the color of the entity.
        """
        return EntityType[self.entity_type.upper()].color


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

        corners = np.vstack((x_corners, y_corners, z_corners))
        rot_test = R.from_euler('z', np.pi/2)
        corners = np.dot(rot_test.as_matrix(), corners)

        return corners
