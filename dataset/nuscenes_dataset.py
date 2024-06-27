from typing import List, Tuple
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import Box
from scipy.spatial.transform import Rotation as R
import numpy as np

from dataset.dataset_interface import DatasetInterface
from dataset.utils.data_clases import Entity, EgoVehicle
from dataset.utils.relationship_extractor import RelationshipExtractor


class NuscenesDataset(DatasetInterface):
    def __init__(self, config: dict) -> None:
        # Ego dimensions: https://forum.nuscenes.org/t/dimensions-of-the-ego-vehicle-used-to-gather-data/550
        self.__ego_vehicle_size__ = np.array([1.73, 1.52, 4.08])    # [width, height, length]
        self.__root_folder__ = config['root_folder']
        self.nusc = NuScenes(version=config['version'], dataroot=self.__root_folder__, verbose=config["verbose"])
        self.sample_token_list = self.load_sample_token_list()
        self.image_token_list, self.image_path_list, self.ego_pose_token_list = self.load_data()
        self.relationship_extractor = RelationshipExtractor()

    def __len__(self) -> int:
        return len(self.sample_token_list)

    def get_image(self, index: int) -> str:
        return self.image_path_list[index]

    def get_ego_vehicle(self, index: int) -> EgoVehicle:
        ego_pose = self.nusc.get('ego_pose', self.ego_pose_token_list[index])
        xyz = np.array(ego_pose['translation'])
        rotation = R.from_quat([
            ego_pose['rotation'][1],  # X
            ego_pose['rotation'][2],  # Y
            ego_pose['rotation'][3],  # Z
            ego_pose['rotation'][0]  # W
        ])
        ego_vehicle = EgoVehicle(xyz, self.__ego_vehicle_size__, rotation)
        return ego_vehicle

    def get_entities(self, index: int) -> List[Entity]:
        annotation = self.nusc.get_sample_data(self.image_token_list[index])
        filtered_annotations = self.filter_annotations(annotation[1])
        entities_list = self.convert_annotations(filtered_annotations)
        return entities_list

    def get_sg_triplets(self, index: int) -> List[Tuple[str, str, str]]:
        sg_triplets = self.relationship_extractor.get_all_relationships(
            self.get_entities(index), self.get_ego_vehicle(index))
        return sg_triplets

    def load_sample_token_list(self) -> List[str]:
        sample_token_list = [s['token'] for s in self.nusc.sample]
        assert len(sample_token_list) > 0, "Error: Database has no samples!"
        return sample_token_list

    def load_data(self) -> Tuple[List[str], List[str], List[str]]:
        image_token_list = []
        image_path_list = []
        ego_pose_token_list = []
        for token in self.sample_token_list:
            sample = self.nusc.get('sample', token)
            image_token_list.append(sample['data']['CAM_FRONT'])
            cam_front_data = self.nusc.get('sample_data', sample['data']['CAM_FRONT'])
            image_path_list.append(self.__root_folder__ + cam_front_data['filename'])
            ego_pose_token_list.append(cam_front_data['ego_pose_token'])
        return (image_token_list, image_path_list, ego_pose_token_list)

    def filter_annotations(self, annotations: List[Box]) -> List[Box]:
        filtered_annotations = []
        for ann in annotations:
            # Remove annotations that are not humans or vehicles
            if ann.name.startswith("human") or ann.name.startswith("vehicle"):
                # Remove annotations with low visibility
                visibility = int(self.nusc.get('sample_annotation', ann.token)['visibility_token'])
                if visibility > 2:
                    filtered_annotations.append(ann)
        return filtered_annotations

    def convert_annotations(self, annotations: List[Box]) -> List[Entity]:
        llmsrp_annotations = []
        for ann in annotations:
            llmsrp_annotations.append(self.box2entity(ann))
        return llmsrp_annotations

    def box2entity(self, ann: Box) -> Entity:
        entity_type = None
        if ann.name.startswith("human"):
            entity_type = "person"
        elif ann.name.startswith("vehicle"):
            vehicle_type = ann.name.split(".")[1]
            if vehicle_type == "construction":
                entity_type = "construction vehicle"
            elif vehicle_type == "emergency":
                entity_type = "emergency vehicle"
            elif vehicle_type == "trailer":
                entity_type = "trailer truck"
            else:
                entity_type = vehicle_type
        else:
            raise ValueError("Error: Unknown entity!")
        # Swap the length and the height of the bounding box
        w, l, h = ann.wlh
        whl = np.array([w, h, l])
        # Convert the quaternion to euler angles
        ypr = R.from_euler("zyx", ann.orientation.yaw_pitch_roll)
        entity = Entity(entity_type, ann.center, whl, ypr)
        return entity
