from typing import List, Tuple
import os
import numpy
from scipy.spatial.transform import Rotation as R
from dataset.utils.plot import ScenePlot
from dataset.dataset_interface import DatasetInterface
from dataset.utils.data_clases import EgoVehicle, Entity
from dataset.utils.relationship_extractor import RelationshipExtractor


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
        self.__root_folder__ = config['root_folder'] if config['root_folder'].endswith(
            '/') else config['root_folder'] + '/'
        self.__image_folder__ = self.__root_folder__ + config['version'] + '/training/image_2/'
        self.list_of_images = self.parse_image_list_from_folder(self.__image_folder__)
        self.__label_folder__ = self.__root_folder__ + 'data_object_label_2/training/label_2/'
        self.__camera_calib_folder__ = self.__root_folder__ + 'data_object_calib/training/calib/'
        self.field_of_view = config['field_of_view']
        self.relationship_extractor = RelationshipExtractor(field_of_view=self.field_of_view)
        self.scene_plot = ScenePlot(field_of_view=self.field_of_view)

    def __len__(self) -> int:
        return len(self.list_of_images)

    def get_bb_triplets(self, index: int) -> List[Tuple[str, List[Tuple[str, str, str]]]]:
        bb_triplets = self.relationship_extractor.get_all_bb_relationships(
            self.get_entities(index), self.get_ego_vehicle(index))
        return bb_triplets

    def get_sg_triplets(self, index: int) -> List[Tuple[str, str, str]]:
        sg_triplets = self.relationship_extractor.get_all_relationships(
            self.get_entities(index), self.get_ego_vehicle(index))
        return sg_triplets

    def get_image(self, index: int) -> str:
        return self.__image_folder__ + self.list_of_images[index] + '.png'

    def get_ego_vehicle(self, index: int) -> EgoVehicle:
        xyz = numpy.array([0, 0, 0])
        rotation = R.from_euler("z", 0)
        ego_vehicle = EgoVehicle(xyz, self.__ego_vehicle_size__, rotation)
        return ego_vehicle

    def get_entities(self, index: int) -> List[Entity]:
        annotations = self.get_annotations_by_index(index)
        filtered_ann = self.get_filtered_ann(annotations)
        list_of_entities = self.convert_annotations(filtered_ann, annotations['camera_intrinsics'])
        return list_of_entities

    def get_filtered_ann(self, annotations) -> List[dict]:
        filtered_ann = []
        for ann in annotations['image_labels']:
            if ann['visibility'] in ['0', '1', '2', '3']:
                filtered_ann.append(ann)
        return filtered_ann

    def convert_annotations(self, image_labels: List[dict], camera_intrinsics) -> List[Entity]:
        annotations = []
        for ann in image_labels:
            annotations.append(self.data2entity(ann, numpy.array(camera_intrinsics)))
        return annotations

    def data2entity(self, ann: dict, camera_intrinsic: numpy.ndarray = numpy.eye(3)) -> Entity:
        entity_type = None
        if ann['name'] == "Pedestrian" or ann['name'] == "Person_sitting":
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
        entity = KittiEntity(entity_type,
                             numpy.array([ann['location'][0], ann['location'][2], ann['location'][1]]).astype(float),
                             whl,
                             ypr,
                             camera_intrinsic,
                             (int(ann['bbox']['left'].split('.')[0]),
                              int(ann['bbox']['top'].split('.')[0]),
                              int(ann['bbox']['right'].split('.')[0]),
                              int(ann['bbox']['bottom'].split('.')[0])))
        return entity

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

    # methods for data processing
    def parse_image_list_from_folder(self, path) -> List[str]:
        list_of_image_file_index = [os.path.splitext(f)[0] for f in os.listdir(path) if f.endswith('.png')]
        return sorted(list_of_image_file_index)

    def get_camera_intrinsic(self, index) -> List[List[float]]:
        # https://mmdetection3d.readthedocs.io/en/v0.17.3/datasets/kitti_det.html
        # (P2: camera2 projection matrix after rectification, an 3x4 array)
        path = self.__camera_calib_folder__ + self.list_of_images[index] + '.txt'
        with open(path, 'r', encoding='utf-8') as file:
            content = file.read()
            l = next(line for line in content.split('\n') if line.startswith('P2'))
            _, value = l.split(':', 1)
            p2 = numpy.array([float(x) for x in value.split()]).reshape(3, 4)
            p2_matrix = p2[:3, :3]
        return p2_matrix.tolist()

    def get_labels_for_image(self, index) -> List[dict]:
        label_file_path = self.__label_folder__ + self.list_of_images[index] + '.txt'
        with open(label_file_path, 'r', encoding='utf-8') as file:
            list_of_labels = [line.strip() for line in file]
        labels = []
        for label in list_of_labels:
            d = label.split()
            if d[0] != 'DontCare':
                obj = {
                    'name': d[0],
                    'truncated': d[1],
                    'visibility': d[2],
                    'bbox': {
                        'left': d[4],
                        'top': d[5],
                        'right': d[6],
                        'bottom': d[7]
                    },
                    'dimensions': {
                        'height': d[8],
                        'width': d[9],
                        'length': d[10]
                    },
                    'location': [d[11], d[12], d[13]],
                    'rotating_y': d[14],
                }
                labels.append(obj)
        return labels

    def get_annotations_by_index(self, index) -> dict:
        obj = {
            'image_serial': self.list_of_images[index],
            'index': index,
            'image_labels': self.get_labels_for_image(index),
            'camera_intrinsics': self.get_camera_intrinsic(index),
        }
        return obj
