from typing import List, Tuple
import tempfile
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from waymo_open_dataset import dataset_pb2 as open_dataset
from dataset.dataset_interface import DatasetInterface
from dataset.utils.data_clases import Entity, EgoVehicle
from dataset.utils.relationship_extractor import RelationshipExtractor
from dataset.utils.plot import ScenePlot


ENTITY_TYPE = {
    1: "CAR",
    2: "PERSON",
    4: "BICYCLE"
}

class WaymoEntity(Entity):
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

class WaymoDataset(DatasetInterface):
    def __init__(self, config: dict) -> None:
        self.__ego_vehicle_size__ = np.array([1.73, 1.52, 4.08])    # [width, height, length]
        self.__root_folder__ = config['root_folder'] if config['root_folder'].endswith(
            '/') else config['root_folder'] + '/'
        self.field_of_view = config['field_of_view']

        # Load the dataset
        self.tfrecord_list = []
        for file in os.listdir(self.__root_folder__):
            # if file.endswith(".tfrecord"):
            if file.endswith("1120_000_1140_000_with_camera_labels.tfrecord"):
                self.tfrecord_list.append(self.__root_folder__ + file)
        self.dataset = tf.data.TFRecordDataset(self.tfrecord_list, compression_type='')
        self.dataset_iter = self.dataset.as_numpy_iterator()

        length=0
        # TODO: Find a better way of doing this!
        for idx,f in enumerate(self.dataset_iter):
            length+=1
        self.length = length
        self.dataset_iter = self.dataset.as_numpy_iterator()

        self.relationship_extractor = RelationshipExtractor(field_of_view=self.field_of_view)
        self.scene_plot = ScenePlot(field_of_view=self.field_of_view)

    def get_image(self, index: int) -> str:
        """Returns image path given an index"""
        # TODO: Find a better way of doing this!
        for i, data in enumerate(self.dataset_iter):
            if i == index:
                frame = open_dataset.Frame()
                frame.ParseFromString(data)
                self.dataset_iter = self.dataset.as_numpy_iterator()
                break

        # Creates a temporary file with the given name and writes data to it.
        temp_dir = tempfile.gettempdir()
        temp_file_path = f"{temp_dir}/waymo_temp_image.jpg"
        
        fig, ax = plt.subplots(1, figsize=(5, 5))
        ax.imshow(tf.image.decode_jpeg(frame.images[0].image))
        plt.grid(False)
        plt.axis('off')
        plt.savefig(temp_file_path, bbox_inches='tight', pad_inches=0, dpi=200)

        return temp_file_path

    def get_ego_vehicle(self, index: int) -> EgoVehicle:
        """Returns ego vehicle given an index"""
        # TODO: Find a better way of doing this!
        for i, data in enumerate(self.dataset_iter):
            if i == index:
                frame = open_dataset.Frame()
                frame.ParseFromString(data)
                self.dataset_iter = self.dataset.as_numpy_iterator()
                break

        xyz = np.array([0.0,0.0,0.0])
        rotation = R.from_euler("z", 0)
        ego_vehicle = EgoVehicle(xyz, self.__ego_vehicle_size__, rotation)
        return ego_vehicle

    def get_entities(self, index: int) -> List[Entity]:
        """Returns list of entities given an index"""
        # TODO: Find a better way of doing this!
        for i, data in enumerate(self.dataset_iter):
            if i == index:
                frame = open_dataset.Frame()
                frame.ParseFromString(data)
                self.dataset_iter = self.dataset.as_numpy_iterator()
                break

        entities = self.frame2entities(frame)
        return entities

    def get_sg_triplets(self, index: int) -> List[Tuple[str, str, str]]:
        """Returns list of scene graph triplets given an index"""
        pass

    def get_bb_triplets(self, index: int) -> List[Tuple[str, List[Tuple[str, str, str]]]]:
        """Returns bounding box and scene graph triplets given an index"""
        pass

    def plot_data_point(self, index: int, out_path: None | str = None) -> None:
        """Plot data point given an index"""
        ego_vehicle = self.get_ego_vehicle(index)
        entities = self.get_entities(index)
        image_path = self.get_image(index)
        self.scene_plot.render_scene(ego_vehicle, entities, image_path, out_path, title=f"Sample {index}")

    def plot_bounding_box(self, index: int, bbs: List[str], entities: List[str] | None = None,
                          out_path: None | str = None) -> None:
        """Plot bounding box given an index, and bounding boxes"""
        pass

    def __len__(self) -> int:
        """Returns length of the dataset"""
        return self.length

    def frame2entities(self, frame: 'open_dataset.Frame') -> List[Entity]:
        entities = []
        ids, boxes = [], {}
        for pll in frame.projected_lidar_labels[0].labels:
            idx = pll.id.split('_FRONT')[0]
            ids.append(idx)
            boxes[idx] = pll.box

        for ll in frame.laser_labels:
            # if ll.most_visible_camera_name == "FRONT" and ll.camera_synced_box.ByteSize():
            # if ll.camera_synced_box.ByteSize():
            if ll.id in ids:
                xyz = np.array([
                    ll.camera_synced_box.center_x,
                    ll.camera_synced_box.center_y,
                    ll.camera_synced_box.center_z
                ])
                # As per https://github.com/waymo-research/waymo-open-dataset/issues/30
                whl = np.array([
                    ll.camera_synced_box.width,  # dim y
                    ll.camera_synced_box.height, # dim z
                    ll.camera_synced_box.length, # dim x
                ])
                rotation = R.from_euler("z", ll.camera_synced_box.heading)
                if ll.type in ENTITY_TYPE:
                    camera_extrinsic = np.array(frame.context.camera_calibrations[0].extrinsic.transform).reshape(4,4)
                    camera_intrinsic = np.array(frame.context.camera_calibrations[0].intrinsic).reshape(3,3)
                    bb = self.box2bb(boxes[ll.id])
                    entity = WaymoEntity(ENTITY_TYPE[ll.type], xyz, whl, rotation, camera_intrinsic, camera_extrinsic,
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
