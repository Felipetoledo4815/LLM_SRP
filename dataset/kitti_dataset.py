import json
from typing import List, Tuple

from dataset.dataset_interface import DatasetInterface
import numpy
from dataset.utils.data_clases import EgoVehicle, Entity
from dataset.utils.relationship_extractor import RelationshipExtractor
from scipy.spatial.transform import Rotation as R


class KittiDataset(DatasetInterface):
    def __init__(self, config: dict) -> None:
        # ego vehicle size is taken from kitti set up page
        self.__ego_vehicle_size__ = numpy.array([1.60, 1.73, 2.71]) # [width, height, length]
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
        d = self.kitti_data[index]['oxts_data']
        rotation = [float(i) for i in self.kitti_data[index]['rotation']]
        R_rect = numpy.array(rotation).reshape(3, 3)
        rotation = R.from_matrix(R_rect)
        quaternion = rotation.as_quat()

        xyz = numpy.array([d['lat'], d['lon'], d['alt']])
        #rotation = R.from_quat([
        #   d['yaw'], d['pitch'], d['roll']
        #])
        ego_vehicle = EgoVehicle(xyz, self.__ego_vehicle_size__, quaternion)
        return ego_vehicle
    def get_entities(self, index: int) -> List[Entity]:
        annotations = self.kitti_data[index]
        list_of_entities = self.convert_annotations(annotations)
        return list_of_entities

    def convert_annotations(self, annotations: {}) -> List[Entity]:
        llmsrp_annotations = []
        for ann in annotations['image_label']:
            llmsrp_annotations.append(self.data2entity(ann, numpy.array(annotations['camera_intrinsics'])))
        return llmsrp_annotations

    def data2entity(self, ann: {}, camera_intrinsic: numpy.ndarray = numpy.eye(3)) -> Entity:
        entity_type = None
        # TODO: Define a mapper for the entity types
        if ann['name'] == "Pedestrian":
            entity_type = "person"
        else:
            entity_type = 'car'
        # Swap the length and the height of the bounding box
        whl = numpy.array([ann['dimensions']['width'], ann['dimensions']['height'], ann['dimensions']['length']])
        whl = whl.astype(float)
        # Convert the quaternion to euler angles
        rotation_y = float(ann['rotating_y'])
        #rotation_matrix = numpy.array([
        #    [numpy.cos(rotation_y), 0, numpy.sin(rotation_y)],
        #    [0, 1, 0],
        #    [-numpy.sin(rotation_y), 0, numpy.cos(rotation_y)]
        #])
        #rotation = R.from_matrix(rotation_matrix)
        #quaternion = rotation.as_quat()  # Returns [x, y, z, w]
        ypr = R.from_euler("zyx", [rotation_y, 0 , 0])
        entity = Entity(entity_type, numpy.array(ann['location']).astype(float), whl, ypr, camera_intrinsic)
        return entity

    def load_kitti_data(self,config: dict):
        root_folder = config['root_folder'] if config['root_folder'].endswith(
            '/') else config['root_folder'] + '/'
        json_file_path = root_folder + 'converted_data.json'
        with open(json_file_path,'r') as file:
            data = json.load(file)
        return data
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

