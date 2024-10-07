import json
from typing import List, Tuple

from dataset.dataset_interface import DatasetInterface
import numpy
from dataset.utils.data_clases import EgoVehicle, Entity
from dataset.utils.relationship_extractor import RelationshipExtractor
from scipy.spatial.transform import Rotation as R


class KittiEntity(Entity):
    def __init__(self, entity_type: str, xyz: numpy.ndarray, whl: numpy.ndarray, rotation: R,
                 camera_intrinsic: numpy.ndarray, bb: Tuple[int, int, int, int]) -> None:
        super().__init__(entity_type, xyz, whl, rotation, camera_intrinsic)
        self.bb = bb

    def get_2d_bounding_box(self) -> Tuple[int, int, int, int]:
        return self.bb


class KittiDataset(DatasetInterface):
    def __init__(self, config: dict) -> None:
        # ego vehicle size is taken from kitti set up page
        self.__ego_vehicle_size__ = numpy.array([1.60, 1.73, 2.71])  # [width, height, length]
        self.__root_folder__ = config['root_folder'] if config['root_folder'].endswith('/') else config['root_folder'] + '/'
        self.field_of_view = config['field_of_view']
        self.kitti_data = self.load_kitti_data(config)
        self.relationship_extractor = RelationshipExtractor(field_of_view=self.field_of_view)
        # need to adopt according to kitti tracking
        return

    def __len__(self) -> int:
        return len(self.kitti_data)

    def get_bb_triplets(self, index: int) -> List[Tuple[str, List[Tuple[str, str, str]]]]:
        bb_triplets = self.relationship_extractor.get_all_bb_relationships(
            self.get_entities(index), self.get_ego_vehicle(index))
        return bb_triplets

    def get_sg_triplets(self, index: int) -> List[Tuple[str, str, str]]:
        sg_triplets = self.relationship_extractor.get_all_relationships(
            self.get_entities(index), self.get_ego_vehicle(index))
        return sg_triplets

    def get_image(self, index: int) -> str:
        return self.kitti_data[index]['image_path']

    def get_ego_vehicle(self, index: int) -> EgoVehicle:
        xyz = numpy.array([0,0,0])
        rotation = R.from_euler("z", 0)
        ego_vehicle = EgoVehicle(xyz, self.__ego_vehicle_size__, rotation)
        return ego_vehicle

    def get_entities(self, index: int) -> List[Entity]:
        annotations = self.kitti_data[index]
        filtered_image_label = self.get_filtered_image_label(annotations)
        list_of_entities = self.convert_annotations(filtered_image_label, annotations['camera_intrinsics'])
        return list_of_entities

    def get_filtered_image_label(self, annotations):
        filtered_image_label = []
        for ann in annotations['image_labels']:
            if ann['visibility'] == '0' or ann['visibility'] == '2' or ann['visibility'] ==  '3' or ann['visibility'] ==  '1':  # ann['visibility'] ==  '0' is considering only fully visible objects or
                filtered_image_label.append(ann)
        return filtered_image_label

    def convert_annotations(self, image_labels: [], camera_intrinsics) -> List[Entity]:
        llmsrp_annotations = []
        for ann in image_labels:
            llmsrp_annotations.append(self.data2entity(ann, numpy.array(camera_intrinsics)))
        return llmsrp_annotations

    def data2entity(self, ann: {}, camera_intrinsic: numpy.ndarray = numpy.eye(3)) -> Entity:
        entity_type = None
        if ann['name'] == "Pedestrian" or ann['name'] ==  "Person_sitting":
            entity_type = "person"
        elif ann['name'] == "Cyclist":
            entity_type = "bicycle"
        elif ann['name'] == "Truck":
            entity_type = "truck"
        else:
            entity_type = "car"
        # Swap the length and the height of the bounding box
        whl = numpy.array([ann['dimensions']['height'], ann['dimensions']['width'], ann['dimensions']['length']])
        whl = whl.astype(float)
        # Convert the quaternion to euler angles
        rotation_y = float(ann['rotating_y'])
        ypr = R.from_euler("z", -rotation_y)
        # ypr = R.from_euler("y", [yaw])
        entity = KittiEntity(entity_type, numpy.array([ann['location'][0], ann['location'][2], ann['location'][1]]).astype(float), whl, ypr, camera_intrinsic,
                             [float(ann['bbox']['left']),
                              float(ann['bbox']['top']), float(ann['bbox']['right']),
                              float(ann['bbox']['bottom'])])
        return entity

    def load_kitti_data(self, config: dict):
        # Need to implement data parsing mechanism from here
        json_file_path = self.__root_folder__ + 'converted_data.json'
        with open(json_file_path, 'r') as file:
            data = json.load(file)
        return data

    def prepare_kitti_data_corpus(self):
        return

    def plot_data_point(self, index: int, out_path: None | str = None) -> None:
        ego_vehicle = self.get_ego_vehicle(index)
        entities = self.get_entities(index)
        image_path = self.get_image(index)
        self.scene_plot.render_scene(ego_vehicle, entities, image_path, out_path, title=f"Sample {index}")

    def plot_bounding_box(self, index: int, bbs: List[str], entities: List[str] | None = None,
                          out_path: None | str = None) -> None:
        image_path = self.get_image(index)
        self.scene_plot.plot_2d_bounding_boxes_from_corners(bbs=bbs, image_path=image_path,
                                                            out_path=out_path, entity_types=entities)
