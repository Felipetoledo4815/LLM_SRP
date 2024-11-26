from collections import defaultdict
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

    def get_lane_relation(self, entities: List[Entity], lanes, ego: EgoVehicle):
        return self.relationship_extractor.get_all_lane_entity_relationship(entities, lanes, ego)

    def get_sg_triplets(self, index: int) -> List[Tuple[str, str, str]]:
        """Returns list of scene graph triplets given an index"""
        sg_triplets = self.relationship_extractor.get_all_relationships(
            self.get_entities(index), self.get_ego_vehicle(index))
        lanes = self.get_lane_line_pairs(self.get_lanes(index))
        lane_relations = self.get_lane_relation(self.get_entities(index), lanes, self.get_ego_vehicle(index))
        sg_triplets.extend(lane_relations)
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
        lane_lines = self.get_lanes(index)
        self.scene_plot.render_scene(ego_vehicle, entities, image_path, out_path, title=f"Sample {index}", lane_lines=lane_lines)

    def plot_bounding_box(self, index: int, bbs: List[str], entities: List[str] | None = None,
                          out_path: None | str = None) -> None:
        image_path = self.get_image(index)
        self.scene_plot.plot_2d_bounding_boxes_from_corners(bbs=bbs, image_path=image_path,
                                                            out_path=out_path, entity_types=entities)

    def __len__(self) -> int:
        """Returns length of the dataset"""
        return len(self.images_path)

    def merge_lane_lines_by_attributes(self, lane_lines, attribute_values = [1, 2, 3, 4]):

        new_lane_lines = [line for line in lane_lines if line.attribute not in attribute_values]

        for attr_value in attribute_values:
            # Filter lane lines with the current attribute value
            filtered_lines = [line for line in lane_lines if line.attribute == attr_value]

            if filtered_lines:
                # Merge xyz arrays from filtered lane lines
                merged_xyz = np.concatenate([line.xyz for line in filtered_lines], axis=1)

                sorted_indices = np.argsort(merged_xyz[1])

                # Rearrange both rows of arr based on the sorted indices of arr[1]
                sorted_xyz = merged_xyz[:, sorted_indices]

                # Create a new LaneLine object
                merged_lane_line = LaneLine(
                    category=filtered_lines[0].category,  # Set category indicating merge
                    xyz=sorted_xyz,
                    track_id=filtered_lines[0].track_id,  # Set a track ID indicating merge
                    attribute=attr_value
                )

                # Append the new merged lane line
                new_lane_lines.append(merged_lane_line)

        return new_lane_lines

    def get_lane_line_pairs(self, sorted_lane_line_list):
        # starting from left
        lanes = []
        number_of_left_lanes = 0

        for idx, line in enumerate(sorted_lane_line_list):
            if idx == len(sorted_lane_line_list) - 1:
                continue

            if sorted_lane_line_list[idx].xyz[0][0] < 0 and sorted_lane_line_list[idx + 1].xyz[0][0] < 0:
                lanes.append({
                    "left_line_of_lane": sorted_lane_line_list[idx],
                    "right_line_of_lane": sorted_lane_line_list[idx + 1],
                    "lane_position": "left+" + str(idx + 1)
                })
                number_of_left_lanes += 1

            elif sorted_lane_line_list[idx].xyz[0][0] > 0 and sorted_lane_line_list[idx+1].xyz[0][0] > 0:
                lanes.append({
                    "left_line_of_lane": sorted_lane_line_list[idx],
                    "right_line_of_lane": sorted_lane_line_list[idx + 1],
                    "lane_position": "right+" + str(idx - number_of_left_lanes)
                })

            elif sorted_lane_line_list[idx].xyz[0][0] < 0 and sorted_lane_line_list[idx + 1].xyz[0][0] > 0:
                lanes.append({
                    "left_line_of_lane": sorted_lane_line_list[idx],
                    "right_line_of_lane": sorted_lane_line_list[idx + 1],
                    "lane_position": "EGO LANE"
                })

            else:
                print("We have an issue as on lane is EGO")

        for item in lanes:
            if "left lane " in item["lane_position"]:
                item["lane_position"] = "left+" + str(number_of_left_lanes)
                number_of_left_lanes -= 1

        for item in lanes:
            if item["left_line_of_lane"].xyz[0][0] > item["right_line_of_lane"].xyz[0][1]:
                temp = item["left_line_of_lane"]
                item["left_line_of_lane"] = item["right_line_of_lane"]
                item["right_line_of_lane"] = temp

        # for item in lanes:
        #     print(f"Lane position {item['lane_position']}, x coordinate left {item['left_line_of_lane'].xyz[0][0]}, "
        #           f"x coordinate right {item['right_line_of_lane'].xyz[0][0]}, left attribute {item['left_line_of_lane'].attribute}, right attribute {item['right_line_of_lane'].attribute}")

        return lanes

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
                if x < 90:
                    filtered_x.append(x)
                    filtered_y.append(y)
                    filtered_z.append(z)

            xyz = np.array([filtered_x,filtered_y,filtered_z])
            # rotation
            rotated_line_points = xyz
            rot = R.from_euler('z', np.pi / 2)
            xyz_transposed = rotated_line_points.T
            xyz_rotated = rot.apply(xyz_transposed)
            xyz = xyz_rotated.T

            min_value_y = np.min(xyz[1])

            if min_value_y > 50:
                continue


            lane_line = LaneLine(line["category"], xyz, line["track_id"], line["attribute"])
            lane_lines.append(lane_line)

        # Sort the LaneLine objects by the x-coordinate of their first point
        # print(f"number of lane line {len(lane_lines)}")
        merged_line = self.merge_lane_lines_by_attributes(lane_lines)
        # print(f"number of merged lane line {len(merged_line)}")
        sorted_lane_lines = sorted(merged_line, key=lambda line: line.xyz[0, 0])

        # # Print the category and tracking ID of the sorted objects
        # for line in sorted_lane_lines:
        #     print(f"Category: {line.category}, Tracking ID: {line.track_id}, Attribute: {line.attribute}, start x: {line.xyz[0][0]}")
        #     if line.xyz[0][0] < 0:
        #         arr = np.insert(line.xyz[0], 0, line.xyz[0][0])
        #         arr1 = np.insert(line.xyz[1], 0, 0)
        #         arr2 = np.insert(line.xyz[2], 0, 0)
        #
        #         line.xyz = np.array([arr,arr1,arr2])
        return sorted_lane_lines

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
