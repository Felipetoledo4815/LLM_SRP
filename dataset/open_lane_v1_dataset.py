from typing import List, Tuple
from functools import lru_cache
import os
import numpy as np
import pickle
from scipy.spatial.transform import Rotation as R
from waymo_open_dataset import dataset_pb2 as open_dataset
from dataset.dataset_interface import DatasetInterface
from dataset.utils.data_clases import Entity, EgoVehicle, LaneLine
from dataset.utils.relationship_extractor import RelationshipExtractor
from dataset.utils.plot import ScenePlot
import json

ENTITY_TYPE = {
    1: "vehicle",
    2: "person",
    3: "sign",
    4: "bicycle"
}


class OpenLaneV1Entity(Entity):
    def __init__(self, entity_type: str, xyz: np.ndarray, whl: np.ndarray, rotation: R, camera_intrinsic: np.ndarray,
                 camera_extrinsic: np.ndarray, bb: Tuple[int, int, int, int]) -> None:
        super().__init__(entity_type, xyz, whl, rotation, camera_intrinsic)
        self.camera_extrinsic = camera_extrinsic
        self.bb = bb

    def corners(self,  whl_factor: float = 1.0) -> np.ndarray:
        corners = super().corners(whl_factor)
        # Rotate the corners to be in East-North-Up (ENU) coordinate system (right-hand rule)
        rot = R.from_euler('z', np.pi/2)
        corners = np.dot(rot.as_matrix(), corners)
        return corners

    def get_2d_bounding_box(self) -> Tuple[int, int, int, int]:
        return self.bb


class OpenLaneV1Dataset(DatasetInterface):
    def __init__(self, config: dict) -> None:
        self.__ego_vehicle_size__ = np.array([1.73, 1.52, 4.08])    # [width, height, length]
        self.__root_folder__ = config['root_folder'] if config['root_folder'].endswith(
            '/') else config['root_folder'] + '/'
        self.field_of_view = config['field_of_view']

        self.images_path, self.labels_path, self.lane_path = self.load_data()

        self.relationship_extractor = RelationshipExtractor(field_of_view=self.field_of_view)
        self.scene_plot = ScenePlot(field_of_view=self.field_of_view)

    def load_data(self) -> Tuple[List[str], List[str], List[str]]:
        images_path = []
        labels_path = []
        lane_path = []
        for file in os.listdir(self.__root_folder__):
            if file.endswith(".jpg"):
                images_path.append(self.__root_folder__ + file)
            if file.endswith(".pkl"):
                labels_path.append(self.__root_folder__ + file)
            if file.endswith(".json"):
                lane_path.append(self.__root_folder__ + file)
        sorted_images_path = sorted(images_path, key=lambda x: int(x.split("/")[-1].split(".")[0].split("_")[-1]))
        sorted_labels_path = sorted(labels_path, key=lambda x: int(x.split("/")[-1].split(".")[0].split("_")[-1]))
        sorted_lane_path = sorted(lane_path, key=lambda x: int(x.split("/")[-1].split(".")[0].split("_")[-1]))
        print(len(sorted_images_path))
        print(len(sorted_labels_path))
        print(len(sorted_lane_path))
        return sorted_images_path, sorted_labels_path, sorted_lane_path

    def get_image(self, index: int) -> str:
        """Returns image path given an index"""
        return self.images_path[index]

    def get_lane_data(self, index: int) -> str:
        """Returns json file path of lane data for given index"""
        return self.lane_path[index]

    def get_ego_vehicle(self, index: int) -> EgoVehicle:
        """Returns ego vehicle given an index. Ego vehicle is always in the center of the coordinate system."""
        xyz = np.array([0.0, 0.0, 0.0])
        rotation = R.from_euler("z", 0)
        ego_vehicle = EgoVehicle(xyz, self.__ego_vehicle_size__, rotation)
        return ego_vehicle


    def get_lanes(self, index: int):
        with open(self.lane_path[index], 'r') as f:
            data = json.load(f) 
        lanes = self.frame2lane(data)
        return lanes

    @lru_cache()
    def get_entities(self, index: int) -> List[Entity]:
        """Returns list of entities given an index"""
        with open(self.labels_path[index], "rb") as f:
            frame = pickle.load(f)
        entities = self.frame2entities(frame)

        # Apply occlusion filter
        not_occluded_entities = []
        for entity in entities:
            if not self.relationship_extractor.is_occluded(entity, entities):
                not_occluded_entities.append(entity)

        return not_occluded_entities

    def get_sg_triplets(self, index: int) -> List[Tuple[str, str, str]]:
        """Returns list of scene graph triplets given an index"""
        sg_triplets = self.relationship_extractor.get_all_relationships(
            self.get_entities(index), self.get_ego_vehicle(index))
        return sg_triplets

    def get_bb_triplets(self, index: int) -> List[Tuple[str, List[Tuple[str, str, str]]]]:
        """Returns bounding box and scene graph triplets given an index"""
        bb_triplets = self.relationship_extractor.get_all_bb_relationships(
            self.get_entities(index), self.get_ego_vehicle(index))
        return bb_triplets

    def plot_data_point(self, index: int, out_path: None | str = None) -> None:
        """Plot data point given an index"""
        ego_vehicle = self.get_ego_vehicle(index)
        entities = self.get_entities(index)
        image_path = self.get_image(index)
        lanes = self.get_lanes(index)
        self.scene_plot.render_scene(ego_vehicle, entities, image_path, out_path, title=f"Sample {index}", lane_lines=lanes)

    def plot_bounding_box(self, index: int, bbs: List[str], entities: List[str] | None = None,
                          out_path: None | str = None) -> None:
        image_path = self.get_image(index)
        self.scene_plot.plot_2d_bounding_boxes_from_corners(bbs=bbs, image_path=image_path,
                                                            out_path=out_path, entity_types=entities)

    def __len__(self) -> int:
        """Returns length of the dataset"""
        return len(self.images_path)

    def frame2lane(self, data):
        number_of_lane_lines = len(data["lane_label"])
        lane_lines = []
        for idx, line in enumerate(data["lane_label"]):
            x_points = line["xyz"][0]
            y_points = line["xyz"][1]
            z_points = line["xyz"][2]

            # Filter the points where x < 100
            filtered_x = []
            filtered_y = []
            filtered_z = []

            for x, y, z in zip(x_points, y_points, z_points):
                if x < 100:
                    filtered_x.append(x)
                    filtered_y.append(y)
                    filtered_z.append(z)

            xyz = np.array([filtered_x,filtered_y,filtered_z])
            print("line number " + str(idx+1) + " number of points are " + str(len(xyz)))
            lane_line = LaneLine("left", xyz)
            lane_lines.append(lane_line)
        return lane_lines

    def frame2entities(self, frame: dict) -> List[Entity]:
        entities = []
        ids, boxes = [], {}
        # Get rid of occluded objects
        filter_available = any([label.num_top_lidar_points_in_box > 0 for label in frame['laser_labels']])
        for pll in frame['projected_lidar_labels'][0].labels:
            idx = pll.id.split('_FRONT')[0]
            ids.append(idx)
            boxes[idx] = pll.box

        for ll in frame['laser_labels']:
            ocluded_object = (filter_available and not ll.num_top_lidar_points_in_box) or (
                not filter_available and not ll.num_lidar_points_in_box)
            if ll.id in ids and not ocluded_object:
                xyz = np.array([
                    ll.camera_synced_box.center_x,
                    ll.camera_synced_box.center_y,
                    ll.camera_synced_box.center_z
                ])
                # As per https://github.com/waymo-research/waymo-open-dataset/issues/30
                whl = np.array([
                    ll.camera_synced_box.width,  # dim y
                    ll.camera_synced_box.height,  # dim z
                    ll.camera_synced_box.length,  # dim x
                ])
                rotation = R.from_euler("z", ll.camera_synced_box.heading)
                if ll.type != 3: # Ignore signs
                    camera_extrinsic = np.array(frame['context'].camera_calibrations[0].extrinsic.transform).reshape(4, 4)
                    camera_intrinsic = np.array(frame['context'].camera_calibrations[0].intrinsic).reshape(3, 3)
                    bb = self.box2bb(boxes[ll.id])
                    entity = OpenLaneV1Entity(ENTITY_TYPE[ll.type], xyz, whl, rotation, camera_intrinsic, camera_extrinsic,
                                         bb)
                    entities.append(entity)
        return entities

    def box2bb(self, box: 'open_dataset.label_pb2.Box') -> Tuple[int, int, int, int]:
        """Converts box to bounding box (bottom left, top right)"""
        center_x = box.center_x * 775 / 1920
        length = box.length * 775 / 1920
        center_y = box.center_y * 516 / 1280
        width = box.width * 516 / 1280
        bb = int(center_x - length/2), int(center_y - width/2), int(center_x + length/2), int(center_y + width/2)
        return bb

    def entity_type_mapper(self, ann:int) -> str:
        # https://medium.com/@lattandreas/2d-detection-on-waymo-open-dataset-f111e760d15b
        return ENTITY_TYPE[ann]
